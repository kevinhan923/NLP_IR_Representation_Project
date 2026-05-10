from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

from .experiment_grid import GridExperiment


class MultiSeedExperiment:
    def __init__(self, base_config: Dict[str, object], seeds: List[int]):
        self.base_config = base_config
        self.seeds = seeds

    def run(self) -> Dict[str, object]:
        all_runs = []
        for seed in self.seeds:
            cfg = json.loads(json.dumps(self.base_config))
            cfg["experiment"]["seed"] = seed
            result = GridExperiment(cfg).run()
            result["seed"] = seed
            all_runs.append(result)

        metrics = ["mrr@10", "recall@10", "precision@10", "hit@10", "ndcg@10"]
        summary = {}
        for side in ["single_vector", "paragraph_level"]:
            summary[side] = {}
            for metric in metrics:
                vals = [r["test_results"][side].get(metric, 0.0) for r in all_runs]
                summary[side][metric] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
        deltas = {}
        for metric in metrics:
            vals = [r["test_results"]["delta_paragraph_minus_single"].get(metric, 0.0) for r in all_runs]
            deltas[metric] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
        return {
            "base_config": self.base_config,
            "seeds": self.seeds,
            "run_count": len(all_runs),
            "summary": summary,
            "delta_summary": deltas,
            "all_runs": all_runs,
        }


def save_json(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
