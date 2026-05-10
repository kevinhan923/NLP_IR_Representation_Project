#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_beir import BeirTransferExperiment, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else (root / "configs" / "beir_transfer.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    result = BeirTransferExperiment(cfg).run()
    out_path = Path(args.output) if args.output else (root / "results" / "beir_transfer_results.json")
    save_json(result, out_path)
    print(f"Saved BEIR transfer results to: {out_path}")


if __name__ == "__main__":
    main()
