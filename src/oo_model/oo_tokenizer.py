"""
BPE Tokenizer custom pour OO-Native.
Vocab 16K orienté domaine système + bas niveau + OO.

Features:
  - Tokens spéciaux OO: [OO:THINK], [OO:ACT], [OO:FEEL], [OO:END], [SAFE], [SYS]
  - Tokens de domaine: [MATH], [CODE], [SYSTEM], [CHAT], [PLAN]
  - Spacer de boucle latente: '=' (token ID fixe = 3)
  - Compatible avec tokenizer.bin format (llm-baremetal)

Usage:
  python oo_tokenizer.py build data/processed/train.jsonl
  python oo_tokenizer.py encode "hello world"
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Special tokens (IDs fixes, ne jamais changer)
# ─────────────────────────────────────────────
SPECIAL_TOKENS = [
    "<unk>",        # 0
    "<bos>",        # 1
    "<eos>",        # 2
    "=",            # 3  — dark loop spacer
    "[OO:THINK]",   # 4
    "[OO:ACT]",     # 5
    "[OO:FEEL]",    # 6
    "[OO:END]",     # 7
    "[SAFE]",       # 8
    "[SYS]",        # 9
    "[MATH]",       # 10
    "[CODE]",       # 11
    "[SYSTEM]",     # 12
    "[CHAT]",       # 13
    "[PLAN]",       # 14
]

VOCAB_SIZE = 16384


class OOTokenizer:
    """
    BPE tokenizer propre au projet OO.
    Construction par paires BPE sur corpus OO.
    Format de sauvegarde: oo_vocab.json (compatible avec futur export C)
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._build_special()

    def _build_special(self):
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.inv_vocab[i] = tok

    def _tokenize_word(self, word: str) -> list[str]:
        """Split word into characters for BPE start."""
        return list(word) + ["</w>"]

    def _get_pairs(self, vocab_freq: dict[tuple, int]) -> Counter:
        pairs: Counter = Counter()
        for word_tuple, freq in vocab_freq.items():
            symbols = list(word_tuple)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: tuple[str, str],
        vocab_freq: dict[tuple, int],
    ) -> dict[tuple, int]:
        new_vocab: dict[tuple, int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word_tuple, freq in vocab_freq.items():
            word = " ".join(word_tuple)
            new_word = word.replace(bigram, replacement)
            new_vocab[tuple(new_word.split())] = freq
        return new_vocab

    def build(self, corpus: list[str], target_vocab: int = VOCAB_SIZE) -> None:
        """Build BPE vocabulary from corpus."""
        # Count word frequencies
        word_freq: Counter = Counter()
        for text in corpus:
            for word in re.findall(r"\S+", text.lower()):
                word_freq[word] += 1

        # Initialize with character-level vocab
        vocab_freq: dict[tuple, int] = {}
        char_vocab: set[str] = set()
        for word, freq in word_freq.items():
            chars = tuple(self._tokenize_word(word))
            vocab_freq[chars] = freq
            char_vocab.update(chars)

        # Add chars to vocab after special tokens
        next_id = len(SPECIAL_TOKENS)
        for ch in sorted(char_vocab):
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                self.inv_vocab[next_id] = ch
                next_id += 1

        # BPE merges
        n_merges = target_vocab - next_id
        print(f"[tokenizer] Building {n_merges} BPE merges (chars={next_id - len(SPECIAL_TOKENS)})...")

        for i in range(n_merges):
            pairs = self._get_pairs(vocab_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=lambda p: pairs[p])
            vocab_freq = self._merge_vocab(best_pair, vocab_freq)
            merged = "".join(best_pair)
            self.merges.append(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                self.inv_vocab[next_id] = merged
                next_id += 1

            if (i + 1) % 500 == 0:
                print(f"[tokenizer] merge {i+1}/{n_merges} vocab_size={next_id}")

        print(f"[tokenizer] Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # Handle special tokens first
        tokens: list[int] = []
        parts = re.split(r"(\[OO:[A-Z]+\]|\[SAFE\]|\[SYS\]|\[MATH\]|\[CODE\]|\[SYSTEM\]|\[CHAT\]|\[PLAN\])", text)
        for part in parts:
            if part in self.vocab:
                tokens.append(self.vocab[part])
            else:
                # BPE encode
                for word in re.findall(r"\S+|\s+", part):
                    word_tokens = self._bpe_encode_word(word)
                    tokens.extend(word_tokens)
        return tokens

    def _bpe_encode_word(self, word: str) -> list[int]:
        if not word.strip():
            # Whitespace: encode as-is
            return [self.vocab.get(ch, 0) for ch in word]
        chars = list(word) + ["</w>"]
        pairs = set((chars[i], chars[i + 1]) for i in range(len(chars) - 1))

        for merge in self.merges:
            if merge not in pairs:
                continue
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == merge:
                    new_chars.append("".join(merge))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            if len(chars) == 1:
                break
            pairs = set((chars[i], chars[i + 1]) for i in range(len(chars) - 1))

        return [self.vocab.get(ch, 0) for ch in chars]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        text = "".join(tokens).replace("</w>", " ").rstrip()
        return text

    def save(self, path: str) -> None:
        out = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": SPECIAL_TOKENS,
        }
        Path(path).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[tokenizer] Saved to {path} ({len(self.vocab)} tokens, {len(self.merges)} merges)")

    @staticmethod
    def load(path: str) -> "OOTokenizer":
        raw = json.loads(Path(path).read_text())
        tok = OOTokenizer()
        tok.vocab = {k: int(v) for k, v in raw["vocab"].items()}
        tok.inv_vocab = {int(v): k for k, v in raw["vocab"].items()}
        tok.merges = [tuple(m) for m in raw["merges"]]
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def cmd_build(corpus_path: str, out_path: str = "data/oo_vocab.json") -> None:
    lines = []
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            lines.append(row.get("instruction", ""))
            lines.append(row.get("response", ""))

    tok = OOTokenizer()
    tok.build(lines, target_vocab=VOCAB_SIZE)
    tok.save(out_path)


def cmd_encode(text: str, vocab_path: str = "data/oo_vocab.json") -> None:
    tok = OOTokenizer.load(vocab_path)
    ids = tok.encode(text)
    print(f"IDs ({len(ids)}): {ids[:50]}{'...' if len(ids) > 50 else ''}")
    print(f"Decoded: {tok.decode(ids)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: oo_tokenizer.py build <corpus.jsonl> [out.json]")
        print("       oo_tokenizer.py encode <text> [vocab.json]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        cmd_build(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/oo_vocab.json")
    elif cmd == "encode":
        cmd_encode(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/oo_vocab.json")
