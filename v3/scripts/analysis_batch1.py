# Run from v3/: ../.venv/bin/python scripts/analysis_batch1.py
"""Batch-1 hardening re-analyses (no GPU, no download).

Four self-contained statistical audits that operate ONLY on already-saved
result JSONs:

  A1. Family-wise correction (Holm + Benjamini-Hochberg) across the primary
      C1 grid (encoder x dataset, agg=max), plus Cohen's d effect sizes.
  A2. Power / minimum-detectable-effect (MDE) for every cell: are the nulls
      powered enough to count as evidence of absence?  (CI-recovered SD and,
      for E5, exact per-query SD from the locality rows.)
  A3. Robustness of the C2 locality correlation: per-dataset Spearman rho,
      pooled dataset-demeaned bootstrap CI, jackknife/drop-influence,
      Kendall's tau, and the independent-encoder (x_) replication.
  A4. Denoised / cross-seed-stable oracle for C3: how much of the +0.029
      oracle headroom is real vs per-query label noise.

Everything is printed with an explicit PASS/FAIL/INFO verdict and dumped to
results/analysis_batch1.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "analysis_batch1.json"

Z_ALPHA = 1.959963985  # two-sided 0.05
Z_POWER80 = 0.8416212  # 80% power
REF_EFFECT = 0.083     # the fiqa-E5 reference penalty magnitude


def load(name: str):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def sd_from_ci(ci_low: float, ci_high: float, n: int) -> float:
    """Recover the paired-difference SD from a (percentile) bootstrap CI."""
    se = (ci_high - ci_low) / (2 * Z_ALPHA)
    return se * math.sqrt(n)


def holm(pvals):
    """Return Holm-adjusted p-values in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    """Return BH-adjusted q-values in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# A1 + A2: assemble the grid, correct, and power
# --------------------------------------------------------------------------- #
GRID_FILES = {
    "TF-IDF": ["controlled_tfidf.json"],
    "MiniLM": ["controlled_dense_minilm.json"],
    "E5": ["controlled_dense_e5.json", "controlled_dense_e5_longdocs.json"],
    "BGE": ["controlled_dense_bge.json"],
}
CEILING = {"trec-covid", "webis-touche2020"}


def collect_grid(agg="max"):
    cells = []
    for enc, files in GRID_FILES.items():
        for fn in files:
            d = load(fn)
            if d is None:
                continue
            for ds, block in d.get("results", {}).items():
                pa = block.get("paragraph_by_agg", {}).get(agg)
                if not pa:
                    continue
                sig = pa["significance_rr10"]
                sd = sd_from_ci(sig["ci_low"], sig["ci_high"], sig["n"])
                cells.append({
                    "encoder": enc, "dataset": ds,
                    "delta": sig["mean_diff"], "p": sig["p_value"], "n": sig["n"],
                    "ci_low": sig["ci_low"], "ci_high": sig["ci_high"],
                    "sd": sd,
                    "cohen_d": sig["mean_diff"] / sd if sd > 0 else float("nan"),
                    "ceiling": ds in CEILING,
                })
    return cells


def per_query_sd_e5():
    """Exact per-query paired-diff SD for E5, per dataset, from locality rows
    (pooled over the 3 seeds)."""
    rows = []
    for fn in ["locality_e5.json", "locality_e5_seed7.json", "locality_e5_seed123.json"]:
        d = load(fn)
        if d:
            rows += d.get("rows", [])
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r["delta_max"])
    out = {}
    for ds, vals in by.items():
        a = np.asarray(vals, float)
        out[ds] = {"n_pooled": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1))}
    return out


def analysis_a1_a2():
    cells = collect_grid("max")
    pvals = [c["p"] for c in cells]
    holm_p = holm(pvals)
    bh_q = benjamini_hochberg(pvals)
    for c, hp, bq in zip(cells, holm_p, bh_q):
        c["holm_p"] = hp
        c["bh_q"] = bq
        # power to detect a true REF_EFFECT given this cell's SD & n
        if c["sd"] > 0:
            ncp = REF_EFFECT * math.sqrt(c["n"]) / c["sd"]
            c["power_vs_ref"] = float(stats.norm.cdf(ncp - Z_ALPHA))
            c["mde80"] = (Z_ALPHA + Z_POWER80) * c["sd"] / math.sqrt(c["n"])
        else:
            c["power_vs_ref"] = float("nan"); c["mde80"] = float("nan")

    pq_sd = per_query_sd_e5()

    print("=" * 88)
    print("A1/A2  PRIMARY GRID (agg=max): correction + power")
    print("=" * 88)
    hdr = f"{'enc':7s} {'dataset':16s} {'delta':>8s} {'raw p':>7s} {'Holm':>7s} {'BH q':>7s} {'d':>6s} {'MDE80':>7s} {'pow@.08':>7s}"
    print(hdr)
    for c in sorted(cells, key=lambda x: (x["encoder"], x["dataset"])):
        flag = "*" if c["bh_q"] < 0.05 else " "
        print(f"{c['encoder']:7s} {c['dataset']:16s} {c['delta']:+8.3f} {c['p']:7.4f} "
              f"{c['holm_p']:7.4f} {c['bh_q']:7.4f}{flag} {c['cohen_d']:+6.2f} "
              f"{c['mde80']:7.3f} {c['power_vs_ref']:7.2f}")

    # headline survival
    def cell(enc, ds):
        return next((c for c in cells if c["encoder"] == enc and c["dataset"] == ds), None)
    headline = [("E5", "fiqa"), ("BGE", "fiqa"), ("E5", "scidocs")]
    surv = {f"{e}:{d}": (cell(e, d) or {}).get("holm_p", 1.0) < 0.05 for e, d in headline}
    a1_pass = all(surv.values())
    print(f"\n  [A1] headline penalties survive Holm<0.05: {surv}  ->  {'PASS' if a1_pass else 'FAIL'}")

    # null power (E5 scifact/nfcorpus): powered if MDE <= REF_EFFECT
    print("\n  [A2] E5 null cells — are they powered to detect a -0.083 effect?")
    a2_notes = {}
    for ds in ["scifact", "nfcorpus"]:
        c = cell("E5", ds)
        exact = pq_sd.get(ds)
        line = f"    {ds:10s} CI-SD MDE80={c['mde80']:.3f} (powered={c['mde80'] <= REF_EFFECT})"
        if exact:
            mde_exact = (Z_ALPHA + Z_POWER80) * exact["sd"] / math.sqrt(exact["n_pooled"])
            line += f" | perQ-SD={exact['sd']:.3f} n={exact['n_pooled']} MDE80={mde_exact:.3f}"
            a2_notes[ds] = {"ci_mde80": c["mde80"], "perq_sd": exact["sd"], "perq_mde80": mde_exact,
                            "powered": bool(mde_exact <= REF_EFFECT)}
        print(line)
    print("\n  [A2] ceiling datasets — confirm underpowered (justifies dropping):")
    for ds in ["trec-covid", "webis-touche2020"]:
        c = cell("E5", ds)
        if c:
            print(f"    {ds:16s} n={c['n']} MDE80={c['mde80']:.3f} (>> {REF_EFFECT} => underpowered={c['mde80'] > REF_EFFECT})")

    return {"cells": cells, "a1_pass": bool(a1_pass), "headline_survival": surv,
            "null_power": a2_notes, "perquery_sd_e5": pq_sd}


# --------------------------------------------------------------------------- #
# A3: locality correlation robustness
# --------------------------------------------------------------------------- #
def load_locality_rows():
    per_seed = {}
    for seed, fn in [(42, "locality_e5.json"), (7, "locality_e5_seed7.json"), (123, "locality_e5_seed123.json")]:
        d = load(fn)
        if d:
            per_seed[seed] = d["rows"]
    return per_seed


def within_demeaned(rows, xcol, ycol="delta_max", min_seg=2):
    """Dataset-demean both x and y, then pooled Spearman."""
    xs, ys = [], []
    by = {}
    for r in rows:
        if r.get("n_segments_min", 0) < min_seg:
            continue
        by.setdefault(r["dataset"], []).append((r[xcol], r[ycol]))
    dm_x, dm_y = [], []
    for ds, pairs in by.items():
        a = np.asarray(pairs, float)
        if a.shape[0] < 5:
            continue
        dm_x += list(a[:, 0] - a[:, 0].mean())
        dm_y += list(a[:, 1] - a[:, 1].mean())
    dm_x, dm_y = np.asarray(dm_x), np.asarray(dm_y)
    return dm_x, dm_y


def analysis_a3():
    per_seed = load_locality_rows()
    print("\n" + "=" * 88)
    print("A3  LOCALITY CORRELATION ROBUSTNESS  (x = max_minus_mean, y = delta_max)")
    print("=" * 88)

    SEL = ["scifact", "nfcorpus", "fiqa"]
    HELD = "scidocs"

    # (a) per-dataset within Spearman, per seed — is the pooled rho fiqa-driven?
    print("\n  (a) per-dataset within Spearman rho (multiseg rows), per seed:")
    per_ds = {}
    for ds in SEL + [HELD]:
        row = []
        for seed, rows in per_seed.items():
            pairs = [(r["max_minus_mean"], r["delta_max"]) for r in rows
                     if r["dataset"] == ds and r.get("n_segments_min", 0) >= 2]
            if len(pairs) >= 8:
                a = np.asarray(pairs, float)
                rho, p = stats.spearmanr(a[:, 0], a[:, 1])
                row.append((seed, rho, p, len(pairs)))
        per_ds[ds] = row
        cells = "  ".join(f"s{seed}:{rho:+.3f}(p={p:.3f},n={n})" for seed, rho, p, n in row)
        print(f"    {ds:10s} {cells}")

    # (b) pooled dataset-demeaned bootstrap CI (per seed, since pools overlap across seeds)
    print("\n  (b) pooled dataset-demeaned Spearman + bootstrap 95% CI (per seed):")
    rng = np.random.default_rng(0)
    boot = {}
    for seed, rows in per_seed.items():
        dm_x, dm_y = within_demeaned(rows, "max_minus_mean")
        rho, p = stats.spearmanr(dm_x, dm_y)
        n = dm_x.size
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            bs.append(stats.spearmanr(dm_x[idx], dm_y[idx]).correlation)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boot[seed] = {"rho": float(rho), "p": float(p), "n": int(n), "ci": [float(lo), float(hi)]}
        print(f"    seed{seed}: rho={rho:+.3f} p={p:.2e} n={n}  boot95=[{lo:+.3f},{hi:+.3f}] "
              f"{'(excludes 0)' if lo > 0 else '(INCLUDES 0)'}")

    # (c) jackknife drop-influence on seed42 pooled demeaned
    print("\n  (c) drop-top-k%-influence (seed42 pooled demeaned) — does rho stay positive?:")
    rows42 = per_seed.get(42, [])
    dm_x, dm_y = within_demeaned(rows42, "max_minus_mean")
    base = stats.spearmanr(dm_x, dm_y).correlation
    n = dm_x.size
    # influence ~ contribution to rank-covariance; approximate by leave-one-out delta
    rx = stats.rankdata(dm_x); ry = stats.rankdata(dm_y)
    prod = (rx - rx.mean()) * (ry - ry.mean())
    infl_order = np.argsort(-np.abs(prod))  # most influential first
    drop_results = {}
    for pct in (1, 2, 5):
        k = max(1, int(n * pct / 100))
        keep = np.ones(n, bool); keep[infl_order[:k]] = False
        rho_k = stats.spearmanr(dm_x[keep], dm_y[keep]).correlation
        drop_results[pct] = float(rho_k)
        print(f"    drop {pct}% (k={k}): rho={rho_k:+.3f}")

    # (d) Kendall tau (robust) + independent-encoder x_ replication (seed42)
    tau = stats.kendalltau(dm_x, dm_y).correlation
    xdm_x, xdm_y = within_demeaned(rows42, "x_max_minus_mean")
    xrho = stats.spearmanr(xdm_x, xdm_y).correlation
    xp = stats.spearmanr(xdm_x, xdm_y).pvalue
    print(f"\n  (d) seed42 Kendall tau={tau:+.3f}; independent-encoder x_max_minus_mean rho={xrho:+.3f} (p={xp:.2e})")

    # verdict
    sel_positive = all(all(rho > 0 for _, rho, _, _ in per_ds[ds]) for ds in SEL)
    ci_excludes0 = all(b["ci"][0] > 0 for b in boot.values())
    drop_ok = all(v > 0 for v in drop_results.values())
    a3_pass = sel_positive and ci_excludes0 and drop_ok
    print(f"\n  [A3] selection-set rho positive in every seed: {sel_positive}; "
          f"bootstrap CI excludes 0 (all seeds): {ci_excludes0}; drop-influence stays +: {drop_ok}")
    print(f"  [A3] VERDICT: {'PASS' if a3_pass else 'FAIL'}")
    return {"per_dataset": {k: [(s, float(r), float(p), n) for s, r, p, n in v] for k, v in per_ds.items()},
            "bootstrap": boot, "drop_influence": drop_results,
            "kendall_tau_seed42": float(tau), "x_rho_seed42": float(xrho),
            "a3_pass": bool(a3_pass)}


# --------------------------------------------------------------------------- #
# A4: denoised / cross-seed-stable oracle
# --------------------------------------------------------------------------- #
def analysis_a4():
    seeds = {}
    for seed in (42, 7, 123):
        d = load(f"adaptive_e5_seed{seed}.json")
        if d:
            seeds[seed] = d["per_query_ndcg10"]
    print("\n" + "=" * 88)
    print("A4  DENOISED / CROSS-SEED-STABLE ORACLE  (is the +0.029 headroom real or noise?)")
    print("=" * 88)

    # naive per-seed means
    def mean_of(pq, sys):
        return float(np.mean(list(pq[sys].values())))

    print("\n  per-seed pooled-test nDCG@10 means:")
    for seed, pq in seeds.items():
        row = {s: mean_of(pq, s) for s in ["document", "paragraph", "global_alpha", "router_soft", "oracle"]}
        print(f"    seed{seed}: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    # winning side (doc vs par) per query, per seed
    win = {}   # qid -> {seed: +1 if par>=doc else -1}
    docpar = {}  # (seed,qid) -> (doc, par)
    for seed, pq in seeds.items():
        for qid in pq["document"]:
            doc = pq["document"][qid]; par = pq["paragraph"][qid]
            docpar[(seed, qid)] = (doc, par)
            win.setdefault(qid, {})[seed] = 1 if par > doc else (-1 if par < doc else 0)

    # cross-seed-stable oracle: for eval seed s, pick side that wins in the majority
    # of the OTHER seeds this qid appears in; skip qids that appear in <2 seeds.
    stable_lift = {}
    naive_lift = {}
    for seed, pq in seeds.items():
        s_gain, o_gain, used = 0.0, 0.0, 0
        base = 0.0
        for qid in pq["document"]:
            doc, par = docpar[(seed, qid)]
            base += max(doc, par)  # naive oracle numerator
            others = [win[qid][s2] for s2 in win[qid] if s2 != seed]
            if not others:
                continue
            vote = sum(others)
            pick_par = vote > 0  # majority of other seeds favored paragraph
            chosen = par if pick_par else doc
            s_gain += chosen - doc  # stable-oracle lift over document
            o_gain += max(doc, par) - doc  # naive-oracle lift over document
            used += 1
        n = len(pq["document"])
        naive_lift[seed] = float(np.mean([max(*docpar[(seed, q)]) - docpar[(seed, q)][0] for q in pq["document"]]))
        stable_lift[seed] = float(s_gain / used) if used else float("nan")
        # note: naive-oracle-over-doc restricted to the same 'used' subset for fair compare
    # label-noise floor: permute doc/par pairing (random side assignment) -> apparent oracle lift
    rng = np.random.default_rng(1)
    noise_floor = {}
    for seed, pq in seeds.items():
        gains = []
        docs = np.array([pq["document"][q] for q in pq["document"]])
        pars = np.array([pq["paragraph"][q] for q in pq["document"]])
        for _ in range(1000):
            flip = rng.random(docs.size) < 0.5
            a = np.where(flip, pars, docs)
            b = np.where(flip, docs, pars)
            gains.append(np.mean(np.maximum(a, b) - a))  # "oracle" over a random baseline
        noise_floor[seed] = float(np.mean(gains))

    # flip fraction: among qids in >=2 seeds, share whose winning side flips
    multi = {q: v for q, v in win.items() if len(v) >= 2}
    flips = sum(1 for q, v in multi.items() if len(set(np.sign(list(v.values())))) > 1)
    flip_frac = flips / len(multi) if multi else float("nan")

    print("\n  naive-oracle vs cross-seed-stable-oracle lift OVER DOCUMENT (per seed):")
    for seed in seeds:
        print(f"    seed{seed}: naive_oracle_lift={naive_lift[seed]:+.4f}  "
              f"stable_oracle_lift={stable_lift[seed]:+.4f}  noise_floor={noise_floor[seed]:+.4f}")
    print(f"\n  cross-seed flip fraction (qids in >=2 seeds, n={len(multi)}): {flip_frac:.3f} "
          f"of queries flip their better side across seeds (unpredictable-by-construction mass)")

    mean_naive = np.mean(list(naive_lift.values()))
    mean_stable = np.mean(list(stable_lift.values()))
    print(f"\n  [A4] mean naive-oracle lift={mean_naive:+.4f}, mean STABLE-oracle lift={mean_stable:+.4f} "
          f"-> stable is {100*mean_stable/mean_naive:.0f}% of naive")
    print("  [A4] INFO: the gap between naive and stable oracle is the label-noise-inflated,"
          " unreachable part of the headroom.")
    return {"naive_lift": naive_lift, "stable_lift": stable_lift, "noise_floor": noise_floor,
            "flip_fraction": float(flip_frac), "mean_naive": float(mean_naive),
            "mean_stable": float(mean_stable)}


def main():
    out = {}
    out["a1_a2"] = analysis_a1_a2()
    out["a3"] = analysis_a3()
    out["a4"] = analysis_a4()

    print("\n" + "#" * 88)
    print("BATCH-1 VERDICT")
    print("#" * 88)
    a1 = out["a1_a2"]["a1_pass"]; a3 = out["a3"]["a3_pass"]
    print(f"  A1 (headline survives FDR/Holm): {'PASS' if a1 else 'FAIL'}")
    print(f"  A3 (locality correlation robust): {'PASS' if a3 else 'FAIL'}")
    print(f"  A2/A4: informational (power of nulls; denoised oracle) — see above.")
    gate = a1 and a3
    print(f"\n  >>> FOUNDATION {'SOLID — safe to proceed to Batch 2/3' if gate else 'AT RISK — stop and reframe before more experiments'} <<<")
    out["gate_pass"] = bool(gate)

    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
