#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_grid import GridExperiment, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else (root / "configs" / ("msmarco_grid_fast.json" if args.fast else "msmarco_grid_full.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = GridExperiment(config).run()
    out_path = Path(args.output) if args.output else (root / "results" / ("grid_fast_results.json" if args.fast else "grid_full_results.json"))
    save_json(result, out_path)
    print(f"Saved grid experiment results to: {out_path}")


if __name__ == "__main__":
    main()
