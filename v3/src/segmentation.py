"""segmentation.py — Split a document into paragraph-level units.

Three strategies (paper Sec. 5.4): natural paragraphs, sentence windows, and
fixed token chunks. Each returns a list of non-empty segment strings; callers
fall back to the whole text when no valid segment is produced.
"""

from __future__ import annotations

import re
from typing import Callable, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def natural_paragraphs(text: str, min_words: int = 5) -> List[str]:
    blocks = [_normalize(b) for b in re.split(r"\n\s*\n", text) if _normalize(b)]
    return [b for b in blocks if len(b.split()) >= min_words]


def sentence_windows(text: str, window: int = 2, stride: int = 1, min_words: int = 5) -> List[str]:
    sents = [_normalize(s) for s in _SENT_SPLIT.split(_normalize(text)) if _normalize(s)]
    if not sents:
        return []
    out: List[str] = []
    for start in range(0, len(sents), stride):
        chunk = sents[start:start + window]
        if len(chunk) < window:
            break
        joined = " ".join(chunk)
        if len(joined.split()) >= min_words:
            out.append(joined)
    return out


def token_chunks(text: str, token_size: int = 128, min_words: int = 5) -> List[str]:
    tokens = _normalize(text).split()
    out: List[str] = []
    for start in range(0, len(tokens), token_size):
        chunk = tokens[start:start + token_size]
        if len(chunk) >= min_words:
            out.append(" ".join(chunk))
    return out


def build_segmenter(cfg) -> Callable[[str], List[str]]:
    """Return text -> list[str] segmenter from a SegmentationConfig.

    Always yields at least one segment (falls back to the whole text), so a
    paragraph-level retriever never sees an empty document.
    """
    mode = cfg.mode

    def segment(text: str) -> List[str]:
        if mode == "natural":
            segs = natural_paragraphs(text, min_words=cfg.min_words)
        elif mode == "sentence_window":
            segs = sentence_windows(text, window=cfg.window, stride=cfg.stride, min_words=cfg.min_words)
        elif mode == "token_chunk":
            segs = token_chunks(text, token_size=cfg.token_size, min_words=cfg.min_words)
        else:
            raise ValueError(f"Unknown segmentation mode: {mode}")
        return segs or [_normalize(text)]

    return segment
