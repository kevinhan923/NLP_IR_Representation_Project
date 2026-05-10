from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .aggregation import AggregationSpec
from .analysis import build_query_level_comparison, grouped_analysis
from .encoders import build_encoder
from .metrics import metric_bundle
from .rerank import TwoStageReranker
from .segmentation import SegmentationSpec, segment_text
from .stage1 import LocalTfidfRetriever
from .types import DocumentExample, QueryExample


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_beir_local_dataset(data_path: Path) -> Tuple[List[DocumentExample], List[QueryExample], Dict[str, Set[str]]]:
    corpus_path = data_path / "corpus.jsonl"
    queries_path = data_path / "queries.jsonl"
    qrels_path = data_path / "qrels" / "test.tsv"

    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            f"Expected BEIR files not found under {data_path}. "
            f"Need corpus.jsonl, queries.jsonl, and qrels/test.tsv"
        )

    docs: List[DocumentExample] = []
    for row in _read_jsonl(corpus_path):
        doc_id = str(row.get("_id", ""))
        title = str(row.get("title", "") or "")
        body = str(row.get("text", "") or "")
        text = (title + "\n\n" + body).strip() if title else body.strip()
        if not doc_id or not text:
            continue
        docs.append(
            DocumentExample(
                doc_id=doc_id,
                title=title,
                text=text,
                metadata={k: v for k, v in row.items() if k not in {"_id", "title", "text"}},
            )
        )

    queries: List[QueryExample] = []
    for row in _read_jsonl(queries_path):
        qid = str(row.get("_id", ""))
        text = str(row.get("text", "") or "").strip()
        if not qid or not text:
            continue
        queries.append(QueryExample(query_id=qid, text=text))

    qrels: Dict[str, Set[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first and line.lower().startswith("query-id"):
                first = False
                continue
            first = False
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            qid = str(parts[0])
            doc_id = str(parts[1]) if len(parts) == 3 else str(parts[2])
            score = 1
            try:
                score = int(parts[-1])
            except Exception:
                score = 1
            if score > 0:
                qrels.setdefault(qid, set()).add(doc_id)

    query_ids_with_qrels = set(qrels.keys())
    queries = [q for q in queries if q.query_id in query_ids_with_qrels]

    return docs, queries, qrels


def save_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


class BeirTransferExperiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _single_vector_run(
        self,
        queries: List[QueryExample],
        stage1_runs,
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for q in queries:
            ranked = stage1_runs.get(q.query_id, [])
            out[q.query_id] = [x.doc_id for x in ranked]
        return out

    def _build_segmentation(self, seg_cfg: dict) -> SegmentationSpec:
        return SegmentationSpec(
            name=seg_cfg["name"],
            mode=seg_cfg["mode"],
            window_size=seg_cfg.get("window_size"),
            stride=seg_cfg.get("stride"),
            token_size=seg_cfg.get("token_size"),
            min_words=seg_cfg.get("min_words", 5),
        )

    def _build_aggregation(self, agg_cfg: dict) -> AggregationSpec:
        return AggregationSpec(
            name=agg_cfg["name"],
            mode=agg_cfg["mode"],
            top_k=agg_cfg.get("top_k"),
            temperature=agg_cfg.get("temperature"),
            threshold=agg_cfg.get("threshold"),
        )

    def run(self) -> dict:
        data_cfg = self.cfg["data"]
        enc_cfg = self.cfg["encoder"]
        exp_cfg = self.cfg["experiment"]
        setting_cfg = self.cfg["selected_setting"]

        beir_root = Path(data_cfg["beir_root"])
        dataset_names = data_cfg["beir_datasets"]

        segmentation = self._build_segmentation(setting_cfg["segmentation"])
        aggregation = self._build_aggregation(setting_cfg["aggregation"])

        all_results = {}

        for ds_name in dataset_names:
            ds_path = beir_root / ds_name
            docs, queries, qrels = load_beir_local_dataset(ds_path)

            fit_text_cap = int(exp_cfg.get("fit_text_cap", 0))
            candidate_pool = int(exp_cfg.get("candidate_pool", 100))
            metric_ks = exp_cfg.get("metric_ks", [10])

            fit_texts = [d.text for d in docs]
            if fit_text_cap and fit_text_cap > 0:
                fit_texts = fit_texts[:fit_text_cap]

            # same encoder family for both single and paragraph reranking
            encoder = build_encoder(enc_cfg["name"], **enc_cfg.get("params", {}))
            encoder.fit(fit_texts)

            stage1 = LocalTfidfRetriever(
                max_features=exp_cfg.get("stage1_max_features", 100000),
                ngram_range=(1, 2),
                stop_words="english",
            ).fit(docs)

            stage1_output = stage1.retrieve(queries, top_k=candidate_pool)

            single_run = self._single_vector_run(queries, stage1_output.runs)

            reranker = TwoStageReranker(
                encoder=encoder,
                segmentation=segmentation,
                aggregation=aggregation,
            )
            paragraph_output = reranker.rerank(queries, stage1_output.runs)
            paragraph_run = {
                qid: [x.doc_id for x in ranked]
                for qid, ranked in paragraph_output.runs.items()
            }

            doc_lookup = {d.doc_id: d for d in docs}
            doc_segment_counts = {
                d.doc_id: len(segment_text(d.text, segmentation))
                for d in docs
            }

            single_metrics = metric_bundle(single_run, qrels, ks=metric_ks)
            paragraph_metrics = metric_bundle(paragraph_run, qrels, ks=metric_ks)

            all_results[ds_name] = {
                "single_vector": single_metrics,
                "paragraph_level": paragraph_metrics,
                "query_level_comparison": build_query_level_comparison(
                    queries=queries,
                    qrels=qrels,
                    single_run=single_run,
                    paragraph_run=paragraph_run,
                    best_paragraphs=paragraph_output.best_paragraphs,
                ),
                "grouped_analysis": grouped_analysis(
                    queries=queries,
                    qrels=qrels,
                    docs=doc_lookup,
                    doc_segment_counts=doc_segment_counts,
                    single_run=single_run,
                    paragraph_run=paragraph_run,
                ),
                "runtime": {
                    "stage1_seconds": stage1_output.latency_seconds,
                    "paragraph_rerank_seconds": paragraph_output.latency_seconds,
                    "avg_segments_per_candidate": paragraph_output.avg_segments_per_candidate,
                    "avg_scored_segments_per_query": paragraph_output.avg_scored_segments_per_query,
                },
            }

        return {
            "config": self.cfg,
            "results": all_results,
        }