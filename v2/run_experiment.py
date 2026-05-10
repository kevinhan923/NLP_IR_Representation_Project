"""
Unified experiment runner for V2.
Usage: python run_experiment.py --config configs/exp1_encoder_ablation.json
"""
import argparse
import json
import os
import sys

# Ensure src/ is importable when running from v2/
sys.path.insert(0, os.path.dirname(__file__))

from src.data import load_msmarco_passage_sample
from src.stage1 import BM25Retriever, DenseRetriever
from src.rerank import BiEncoderReranker, CrossEncoderReranker
from src.segmentation import StructuralSegmenter, SentenceWindowSegmenter
from src.pipeline import TwoStagePipeline
from src.analysis import error_analysis


def build_retriever(name: str):
    if name == "bm25":
        return BM25Retriever()
    elif name.startswith("dense:"):
        model = name.split(":", 1)[1]
        return DenseRetriever(model_name=model)
    else:
        raise ValueError(f"Unknown retriever: {name}")


def build_reranker(cfg: dict):
    seg_name = cfg.get("segmenter", "structural")
    segmenter = StructuralSegmenter() if seg_name == "structural" else SentenceWindowSegmenter()
    agg = cfg.get("aggregation", "max")
    if cfg["reranker"] == "biencoder":
        return BiEncoderReranker(model_name=cfg["reranker_model"], segmenter=segmenter, aggregation=agg)
    elif cfg["reranker"] == "crossencoder":
        return CrossEncoderReranker(model_name=cfg["reranker_model"], segmenter=segmenter, aggregation=agg)
    else:
        raise ValueError(f"Unknown reranker: {cfg['reranker']}")


def run(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\n=== {cfg['name']} ===")
    print(f"Description: {cfg['description']}\n")

    data = load_msmarco_passage_sample(
        n_docs=cfg["n_docs"], n_queries=cfg["n_queries"], seed=cfg["seed"]
    )
    passages = data["passages"]
    queries = data["queries"]
    qrels = data["qrels"]

    all_results = {}
    for stage1_name in cfg["stage1_options"]:
        print(f"--- Stage 1: {stage1_name} ---")
        retriever = build_retriever(stage1_name)
        reranker = build_reranker(cfg)
        pipeline = TwoStagePipeline(
            retriever=retriever,
            reranker=reranker,
            top_k_stage1=cfg["top_k_stage1"],
            top_k_stage2=cfg["top_k_stage2"],
        )
        pipeline.index(passages)
        metrics = pipeline.evaluate(queries, qrels)

        print(f"  Single MRR@10:     {metrics['single_mrr10']:.4f}")
        print(f"  Single NDCG@10:    {metrics['single_ndcg10']:.4f}")
        print(f"  Single Recall@100: {metrics['single_recall100']:.4f}")
        print(f"  Para   MRR@10:     {metrics['para_mrr10']:.4f}")
        print(f"  Para   NDCG@10:    {metrics['para_ndcg10']:.4f}")
        print(f"  Para   Recall@100: {metrics['para_recall100']:.4f}")
        print(f"  Delta  MRR@10:     {metrics['delta_mrr10']:+.4f}")

        if cfg.get("run_error_analysis"):
            analysis = error_analysis(
                queries=queries,
                single_rankings=metrics["single_rankings"],
                para_rankings=metrics["para_rankings"],
                qrels=qrels,
                passages=passages,
            )
            print(f"\n  Error Analysis:")
            print(f"    Overall delta:        {analysis['overall_delta']:+.4f}")
            print(f"    Para wins / loses:    {analysis['n_queries_para_wins']} / {analysis['n_queries_para_loses']}")
            print(f"    By query length:      {analysis['by_query_length']}")
            print(f"    By doc length:        {analysis['by_doc_length']}")

        all_results[stage1_name] = metrics

    # Save results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{cfg['name']}_results.json")
    with open(out_path, "w") as f:
        save = {k: {m: v for m, v in r.items() if m not in ("single_rankings", "para_rankings")}
                for k, r in all_results.items()}
        json.dump({"config": cfg, "results": save}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
