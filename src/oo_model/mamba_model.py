"""
OO Mamba SSM Model — latent reasoning engine.

Architecture:
  - Base: Mamba SSM (x_proj + dt_proj are the trained layers)
  - HaltingHead: MLP that predicts P(halt) given hidden state + loop position
  - Dark loop: '=' tokens used as silent recurrence cycles before surface output

Maps to OO cognitive loop:
  dark_loop (Mamba recurrence) → D+ judgment → action
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class HaltingHead(nn.Module):
    """
    Predicts P(halt) from [h_d_model | loop_pos/max_loops].
    Input dim = d_model + 1 (normalized loop position scalar).
    Uses fractional ramp labels (not binary) to avoid representational collapse.
    """

    def __init__(self, d_input: int = 769, hidden_dims: list[int] = (512, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_input
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(0.1)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, loop_pos: torch.Tensor) -> torch.Tensor:
        """
        h        : (batch, d_model) — last hidden state
        loop_pos : (batch,) — normalized loop position [0, 1]
        Returns  : (batch,) — P(halt) in [0, 1]
        """
        x = torch.cat([h, loop_pos.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


class OOMambaEngine(nn.Module):
    """
    OO Mamba latent reasoning engine wrapper.
    Wraps a HuggingFace Mamba model + HaltingHead for use in training and inference.
    """

    DOMAIN_MAX: dict[str, int] = {
        "chat": 5,
        "math": 20,
        "code": 35,
        "tool": 8,
        "system": 12,
    }

    def __init__(
        self,
        base_model_name: str = "state-spaces/mamba-130m-hf",
        halt_threshold: float = 0.7,
        d_model: int = 768,
    ):
        super().__init__()
        self.halt_threshold = halt_threshold
        self.d_model = d_model

        # Load base Mamba model
        from transformers import AutoModelForCausalLM
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Freeze all params except x_proj, dt_proj, embed_tokens
        for name, param in self.backbone.named_parameters():
            trainable = any(k in name for k in ("x_proj", "dt_proj", "embed_tokens"))
            param.requires_grad = trainable

        # HaltingHead: input = d_model + 1 (loop pos scalar)
        self.halting_head = HaltingHead(d_input=d_model + 1, hidden_dims=[512, 64])

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Standard forward pass for SFT training."""
        out = self.backbone(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        return {
            "loss": out.loss,
            "logits": out.logits,
            "hidden_states": out.hidden_states,
        }

    @torch.no_grad()
    def generate_latent(
        self,
        tokenizer,
        prompt: str,
        domain: str = "math",
        max_new_tokens: int = 120,
        device: str = "cuda",
    ) -> str:
        """
        Latent reasoning inference:
        1. Iteratively append '=' tokens (dark loops)
        2. HaltingHead decides when to stop looping
        3. Generate surface tokens after halt
        """
        max_loops = self.DOMAIN_MAX.get(domain, 20)
        self.eval()

        for lp in range(50):
            loop_prompt = prompt + "=" * lp
            toks = tokenizer(
                loop_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            out = self.backbone(**toks, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float()
            lp_norm = torch.tensor([lp / max_loops], dtype=torch.float32, device=device)
            p_halt = self.halting_head(h.unsqueeze(0), lp_norm.unsqueeze(0)).item()

            if p_halt >= self.halt_threshold:
                break

        gen_ids = self.backbone.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
        return tokenizer.decode(
            gen_ids[0][toks["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def count_trainable_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
