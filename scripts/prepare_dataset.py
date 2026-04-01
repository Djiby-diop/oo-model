from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = [
    "boot_recovery",
    "operator_command",
    "journal_memory",
    "policy_safety",
    "system_reasoning",
]


def ensure_file(path: Path, sample: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the first OO dataset scaffold.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    sample = {
        "id": "seed-0001",
        "family": "operator_command",
        "source": "manual-seed",
        "input": "Explain the current continuity state.",
        "target": "Report the last known continuity epoch, mode, and recovery reason.",
        "context": {"repo": "oo-system", "component": "interface", "risk": "low"},
        "tags": ["continuity", "command"],
        "quality": 1.0,
    }

    for split in ("train", "valid", "test", "eval_oo"):
        ensure_file(args.output / f"{split}.jsonl", sample)

    manifest = {
        "input_dir": str(args.input),
        "output_dir": str(args.output),
        "families": FAMILIES,
        "status": "scaffold-created",
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("oo-model dataset scaffold ready")


if __name__ == "__main__":
    main()
