from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import ir_datasets
from tqdm import tqdm

from .types import DocumentExample, QueryExample


def _safe_attr(obj, names: Sequence[str], default: str = "") -> str:
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is None:
                continue
            return str(val)
    return default


def msmarco_load_documents(dataset_name: str, max_docs: Optional[int] = None) -> List[DocumentExample]:
    ds = ir_datasets.load(dataset_name)
    docs: List[DocumentExample] = []
    it = ds.docs_iter()
    count = 0
    for doc in tqdm(it, desc=f"Loading docs from {dataset_name}"):
        doc_id = _safe_attr(doc, ["doc_id", "id"])
        title = _safe_attr(doc, ["title"])
        body = _safe_attr(doc, ["body", "text", "contents"])
        url = _safe_attr(doc, ["url"])
        if not doc_id or not body:
            continue
        text = (title + "\n\n" + body).strip() if title else body
        docs.append(DocumentExample(doc_id=doc_id, title=title, text=text, metadata={"url": url}))
        count += 1
        if max_docs is not None and count >= max_docs:
            break
    return docs


def msmarco_load_queries(dataset_name: str, max_queries: Optional[int] = None) -> List[QueryExample]:
    ds = ir_datasets.load(dataset_name)
    queries: List[QueryExample] = []
    count = 0
    for q in tqdm(ds.queries_iter(), desc=f"Loading queries from {dataset_name}"):
        qid = _safe_attr(q, ["query_id", "doc_id", "id"])
        text = _safe_attr(q, ["text", "query"])
        if not qid or not text:
            continue
        queries.append(QueryExample(query_id=qid, text=text))
        count += 1
        if max_queries is not None and count >= max_queries:
            break
    return queries


def msmarco_load_qrels(dataset_name: str, positive_only: bool = True, max_queries: Optional[int] = None) -> Dict[str, Set[str]]:
    ds = ir_datasets.load(dataset_name)
    qrels: Dict[str, Set[str]] = {}
    for qr in tqdm(ds.qrels_iter(), desc=f"Loading qrels from {dataset_name}"):
        rel = int(getattr(qr, "relevance", 0))
        if positive_only and rel <= 0:
            continue
        qid = str(getattr(qr, "query_id"))
        doc_id = str(getattr(qr, "doc_id"))
        qrels.setdefault(qid, set()).add(doc_id)
    if max_queries is not None:
        keep = set(list(qrels.keys())[:max_queries])
        qrels = {qid: docs for qid, docs in qrels.items() if qid in keep}
    return qrels


def build_doc_lookup(docs: List[DocumentExample]) -> Dict[str, DocumentExample]:
    return {d.doc_id: d for d in docs}


def export_jsonl_documents(docs: List[DocumentExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "metadata": doc.metadata}, ensure_ascii=False) + "\n")


def export_jsonl_queries(queries: List[QueryExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query_id": q.query_id, "text": q.text}, ensure_ascii=False) + "\n")


def export_qrels_tsv(qrels: Dict[str, Set[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for doc_id in docs:
                f.write(f"{qid}\t0\t{doc_id}\t1\n")
