#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_multiseed import MultiSeedExperiment, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else (root / "configs" / ("msmarco_multiseed_fast.json" if args.fast else "msmarco_multiseed_full.json"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_cfg_path = root / cfg["base_config"]
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    result = MultiSeedExperiment(base_cfg, cfg["seeds"]).run()
    out_path = Path(args.output) if args.output else (root / "results" / ("multiseed_fast_results.json" if args.fast else "multiseed_full_results.json"))
    save_json(result, out_path)
    print(f"Saved multi-seed experiment results to: {out_path}")


if __name__ == "__main__":
    main()
