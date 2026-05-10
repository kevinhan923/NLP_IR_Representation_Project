from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryExample:
    query_id: str
    text: str


@dataclass
class DocumentExample:
    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateResult:
    doc_id: str
    score: float
    text: Optional[str] = None
    title: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class QueryRun:
    query_id: str
    query_text: str
    ranked: List[CandidateResult]


@dataclass
class ComparisonRow:
    query_id: str
    query: str
    single_rr: float
    paragraph_rr: float
    rr_gain: float
    single_top_doc: str
    paragraph_top_doc: str
    paragraph_top_doc_score: float
    paragraph_top_doc_is_relevant: bool
    best_para_id: int
    best_para_score: float
    best_para_text: str
