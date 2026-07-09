"""summarize.py — Aggregate all results/*.json into one encoder x dataset table.

Reads every controlled-comparison result file and prints, per encoder and
dataset, the document- vs paragraph-level RR@10 with the paired significance
verdict. This is the Contribution-1 view: does granularity matter, and does the
answer depend on the encoder?

Usage:  python scripts/summarize.py
"""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def verdict(delta: float, p: float) -> str:
    if p >= 0.05:
        return "no sig. diff"
    return "paragraph WINS" if delta > 0 else "document WINS"


def main() -> None:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        print("no results/*.json yet")
        return

    print(f"{'encoder':<18}{'dataset':<10}{'doc RR@10':>10}{'par RR@10':>11}"
          f"{'ΔRR@10':>9}{'p':>7}   verdict (agg=max)")
    print("-" * 86)

    tally = {}  # encoder -> [wins, ties, losses]
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        enc = data.get("encoder", "?")
        enc_label = os.path.basename(fp).replace("controlled_", "").replace(".json", "")
        for ds, r in data["results"].items():
            doc = r["document"]["RR@10"]
            mx = r["paragraph_by_agg"].get("max")
            if not mx:
                continue
            par = mx["paragraph"]["RR@10"]
            delta = mx["delta_rr10"]
            p = mx["significance_rr10"]["p_value"]
            v = verdict(delta, p)
            print(f"{enc_label:<18}{ds:<10}{doc:>10.4f}{par:>11.4f}{delta:>+9.4f}{p:>7.3f}   {v}")
            t = tally.setdefault(enc_label, [0, 0, 0])
            t[0 if v.startswith("paragraph") else (2 if v.startswith("document") else 1)] += 1
        print("-" * 86)

    print("\nPer-encoder tally (agg=max, across datasets):")
    for enc_label, (w, t, l) in tally.items():
        print(f"  {enc_label:<18} paragraph wins {w} / ties {t} / document wins {l}")


if __name__ == "__main__":
    main()
