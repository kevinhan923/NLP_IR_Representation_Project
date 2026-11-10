"""export_msmarco_beir.py — export TREC DL 2019 document as a BEIR-layout dataset.

Why this exists
---------------
The paper's grid and multi-seed experiments built their pool with
``v1/src/datasets.py::msmarco_load_documents``, which takes the FIRST 50,000
documents in corpus order. Measured against the TREC DL 2019 judgments that
leaves:

    distinct relevant docs        6,524
    relevant docs inside the pool   110   (1.69%)
    queries with >=1 rel in pool   30 / 43
    mean relevant-in-pool/query    2.58

so 13 of 43 queries scored zero for both systems by construction, and Recall@10
was computed against a denominator ~60x larger than anything reachable.

This script writes the same 50,000-document pool size but forces every relevant
document in first, then fills with distractors — the guarantee that
``v3/src/data.py::load_beir_sample`` already provides for the BEIR datasets.
Writing BEIR layout means v3 needs no code change to run on it.

Requires ir_datasets, which lives in the anaconda environment, NOT in .venv:

    ~/anaconda3/bin/python scripts/export_msmarco_beir.py
"""
from __future__ import annotations

import json
import os
import random

import ir_datasets

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "v1", "data", "beir", "msmarco-doc")
DATASET = "msmarco-document/trec-dl-2019"
POOL, SEED = 50000, 42


def main() -> None:
    os.makedirs(os.path.join(OUT, "qrels"), exist_ok=True)
    ds = ir_datasets.load(DATASET)

    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        r = int(qr.relevance)
        if r > 0:
            qrels.setdefault(str(qr.query_id), {})[str(qr.doc_id)] = r
    positives = {d for v in qrels.values() for d in v}
    print(f"queries={len(qrels)}  positives={len(positives)}", flush=True)

    qtext = {str(q.query_id): q.text for q in ds.queries_iter()}
    with open(f"{OUT}/queries.jsonl", "w", encoding="utf-8") as f:
        for qid in qrels:
            f.write(json.dumps({"_id": qid, "text": qtext[qid]}) + "\n")

    with open(f"{OUT}/qrels/test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, rels in qrels.items():
            for did, r in rels.items():
                f.write(f"{qid}\t{did}\t{r}\n")

    need = POOL - len(positives)
    kept: list[dict] = []
    distract: list[dict] = []
    scanned = 0
    for d in ds.docs_iter():
        did = str(d.doc_id)
        body = (d.body or "").strip()
        if not body:
            continue
        rec = {"_id": did, "title": (d.title or "").strip(), "text": body}
        if did in positives:
            kept.append(rec)
        elif len(distract) < need:
            distract.append(rec)
        scanned += 1
        if len(kept) == len(positives) and len(distract) >= need:
            break
    print(f"scanned={scanned}  positives_found={len(kept)}  "
          f"distractors={len(distract)}", flush=True)

    pool = kept + distract
    random.Random(SEED).shuffle(pool)
    with open(f"{OUT}/corpus.jsonl", "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(pool)} docs to {OUT}/corpus.jsonl", flush=True)


if __name__ == "__main__":
    main()
