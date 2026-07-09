"""run_controlled.py — CLI entry for the controlled comparison (M0).

Usage:
    python scripts/run_controlled.py --config configs/controlled_tfidf.yaml
"""

import argparse
import json
import os
import sys

# Make src importable when run from v3/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_experiment(cfg)

    for ds, r in out["results"].items():
        d = r["document"]
        print(f"\n=== {ds}  (docs={r['n_docs']}, judged_queries={r['n_queries_judged']}) ===")
        print(f"  document    RR@10={d['RR@10']:.4f}  nDCG@10={d['nDCG@10']:.4f}  R@100={d['R@100']:.4f}")
        for agg, a in r["paragraph_by_agg"].items():
            p, s = a["paragraph"], a["significance_rr10"]
            star = "*" if s["p_value"] < 0.05 else " "
            print(
                f"  paragraph[{agg:>10}] RR@10={p['RR@10']:.4f} nDCG@10={p['nDCG@10']:.4f}"
                f"  ΔRR@10={a['delta_rr10']:+.4f} p={s['p_value']:.3f}{star}"
                f" CI=[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]"
            )

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{cfg.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print("(* = paragraph vs document RR@10 difference significant at p<0.05)")


if __name__ == "__main__":
    main()
