"""
OO Native SSM — architecture propre au projet OO.
From scratch. Bare-metal first. No external pretrained weights.

Architecture:
  - OO-SSM: State Space Model minimal (inspiré Mamba mais custom)
  - 3 tetes OO-spécifiques:
      * PolicyHead   : compatibilité D+ (classe d'action autorisée)
      * PressureHead : niveau de pression mémoire [0=OK → 1=DYING]
      * HaltHead     : latent loop halting (boucles =)
  - Tokenizer BPE custom 16K vocab (domaine système + bas niveau)
  - Export direct vers format binaire OONV (compatible ssm_infer.c)

Design principles:
  - Aucune dépendance HuggingFace en inférence (export standalone)
  - Paramètres ~16M → tient dans 128MB UEFI (q8_0)
  - Séquence de pensée OO: [OO:THINK] → dark_loops → [OO:ACT] → output
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class OONativeConfig:
    vocab_size:     int   = 16384
    d_model:        int   = 512
    n_layer:        int   = 12
    d_state:        int   = 16
    d_conv:         int   = 4
    expand:         int   = 2
    context_length: int   = 1024
    dropout:        float = 0.0

    # OO special token IDs (assigned after tokenizer build)
    tok_loop:       int   = 3    # '='
    tok_think:      int   = 4    # [OO:THINK]
    tok_act:        int   = 5    # [OO:ACT]
    tok_feel:       int   = 6    # [OO:FEEL]
    tok_end:        int   = 7    # [OO:END]
    tok_safe:       int   = 8    # [SAFE]

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.d_model / 16)


# ─────────────────────────────────────────────
# OO-SSM Block (Mamba-style, custom)
# ─────────────────────────────────────────────

class OOSSMBlock(nn.Module):
    """
    Single SSM block. Implements selective state-space recurrence.
    Equivalent to one Mamba layer but stripped for bare-metal export.
    """

    def __init__(self, cfg: OONativeConfig):
        super().__init__()
        d = cfg.d_model
        di = cfg.d_inner
        dt_rank = cfg.dt_rank
        ds = cfg.d_state

        self.norm = nn.LayerNorm(d)

        # Input projection (x and z branches)
        self.in_proj = nn.Linear(d, di * 2, bias=False)

        # Conv1d for local context
        self.conv1d = nn.Conv1d(di, di, kernel_size=cfg.d_conv, padding=cfg.d_conv - 1, groups=di, bias=True)

        # SSM projections
        self.x_proj  = nn.Linear(di, dt_rank + ds * 2, bias=False)  # dt + B + C
        self.dt_proj = nn.Linear(dt_rank, di, bias=True)

        # SSM parameters (fixed A, trainable log_A)
        A = torch.arange(1, ds + 1, dtype=torch.float32).unsqueeze(0).expand(di, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))

        # Output projection
        self.out_proj = nn.Linear(di, d, bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        # Dual branch
        xz = self.in_proj(x)                          # (B, L, 2*di)
        xz = xz.transpose(1, 2)                       # (B, 2*di, L)
        x_branch, z_branch = xz.chunk(2, dim=1)       # each (B, di, L)

        # Conv1d local context
        x_branch = self.act(self.conv1d(x_branch)[..., :L])

        # SSM selective scan (parallel prefix-sum — GPU-efficient)
        x_flat = x_branch.transpose(1, 2)             # (B, L, di)
        dt_BC = self.x_proj(x_flat)                   # (B, L, dt_rank + 2*ds)
        dt, B_ssm, C_ssm = dt_BC.split([self.dt_proj.in_features,
                                         self.A_log.shape[1],
                                         self.A_log.shape[1]], dim=-1)
        dt = F.softplus(self.dt_proj(dt))              # (B, L, di)
        A = -torch.exp(self.A_log.float())             # (di, ds)

        # Discretize: dA(t) = exp(dt * A), dB(t) = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)           # (B, L, di, ds)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)     # (B, L, di, ds)

        # Parallel SSM scan via cumulative product (vectorized, no Python loop)
        # y(t) = sum_s C(t,s) * h(t,s)
        # h(t) = dA(t)*h(t-1) + dB(t)*u(t)
        u = x_flat.unsqueeze(-1)                       # (B, L, di, 1)
        Bu = (dB * u).float()                          # (B, L, di, ds)

        # Compute cumulative dA product for each position (log-space for stability)
        log_dA = torch.log(dA.clamp(min=1e-8))        # (B, L, di, ds)
        cum_log_dA = torch.cumsum(log_dA, dim=1)       # (B, L, di, ds)
        cum_dA = torch.exp(cum_log_dA)                 # (B, L, di, ds)

        # h(t) = sum_{s<=t} dA(s+1..t) * dB(s) * u(s)
        # Efficient: h(t) = cum_dA(t) * sum_{s<=t} Bu(s) / cum_dA(s)
        safe_cum_dA = cum_dA.clamp(min=1e-8)
        scaled_Bu = Bu / safe_cum_dA                   # (B, L, di, ds)
        cum_scaled_Bu = torch.cumsum(scaled_Bu, dim=1) # (B, L, di, ds)
        h = cum_dA * cum_scaled_Bu                     # (B, L, di, ds)

        # y(t) = sum_s C(t,s) * h(t,s)
        y = (h * C_ssm.unsqueeze(2)).sum(-1)           # (B, L, di)
        y = y + x_flat * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z branch
        y = y * self.act(z_branch.transpose(1, 2))

        return self.out_proj(y) + residual


# ─────────────────────────────────────────────
# OO-Specific Heads
# ─────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    D+ policy compatibility head.
    Predicts action class from hidden state.
    32 classes = D+ action vocabulary.
    """
    N_ACTIONS = 32

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.N_ACTIONS),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → (B, N_ACTIONS) logits"""
        return self.net(h)


class PressureHead(nn.Module):
    """
    Memory pressure prediction head.
    Output: scalar [0=OK, 1=DYING] — mirrors OO pressure system.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → (B,) pressure scalar"""
        return self.net(h).squeeze(-1)


class HaltHead(nn.Module):
    """
    Latent loop halting head (OO version).
    Same design as batteryphil's HaltingHead but with d_model from OO-Native.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, loop_pos: torch.Tensor) -> torch.Tensor:
        """
        h        : (B, d_model)
        loop_pos : (B,) normalized [0,1]
        → (B,) P(halt)
        """
        x = torch.cat([h, loop_pos.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# OO Native Model
# ─────────────────────────────────────────────

class OONativeModel(nn.Module):
    """
    OO Native SSM model — the sovereign intelligence core.

    Properties:
    - From scratch (no pretrained weights dependency)
    - ~16M parameters at default config
    - 3 OO-specific heads: policy, pressure, halt
    - Designed for bare-metal export via export_oo_native.py
    - Inference loop: [OO:THINK] + dark_loops → [OO:ACT] → tokens
    """

    def __init__(self, cfg: Optional[OONativeConfig] = None):
        super().__init__()
        self.cfg = cfg or OONativeConfig()
        c = self.cfg

        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_embed  = nn.Embedding(c.context_length, c.d_model)

        self.layers = nn.ModuleList([OOSSMBlock(c) for _ in range(c.n_layer)])
        self.norm_out = nn.LayerNorm(c.d_model)

        # Language modeling head (shared weights with embedding)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        # OO-specific heads
        self.policy_head   = PolicyHead(c.d_model)
        self.pressure_head = PressureHead(c.d_model)
        self.halt_head     = HaltHead(c.d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_oo_heads: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        input_ids : (B, L)
        labels    : (B, L) — for LM loss (CrossEntropy, shift by 1)
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)

        h = self.embedding(input_ids) + self.pos_embed(pos).unsqueeze(0)

        for layer in self.layers:
            h = layer(h)

        h = self.norm_out(h)
        logits = self.lm_head(h)  # (B, L, vocab_size)

        result: dict = {"logits": logits, "hidden": h}

        # LM loss
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.cfg.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        # OO heads on last token
        if return_oo_heads:
            last_h = h[:, -1, :]  # (B, d_model)
            result["policy_logits"]  = self.policy_head(last_h)
            result["pressure"]       = self.pressure_head(last_h)

        return result

    def predict_halt(self, h_last: torch.Tensor, loop_pos: torch.Tensor) -> torch.Tensor:
        """Check if latent loop should halt. h_last: (B, d_model)"""
        return self.halt_head(h_last, loop_pos)

    def count_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @staticmethod
    def from_config(path: str) -> "OONativeModel":
        import json
        from pathlib import Path
        raw = json.loads(Path(path).read_text())
        arch = raw["architecture"]
        cfg = OONativeConfig(
            vocab_size     = arch["vocab_size"],
            d_model        = arch["d_model"],
            n_layer        = arch["n_layer"],
            d_state        = arch["d_state"],
            d_conv         = arch["d_conv"],
            expand         = arch["expand"],
            context_length = arch["context_length"],
        )
        return OONativeModel(cfg)
