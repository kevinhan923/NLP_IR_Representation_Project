from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


@dataclass(frozen=True)
class SegmentationSpec:
    name: str
    mode: str
    window_size: int | None = None
    stride: int | None = None
    token_size: int | None = None
    min_words: int = 5


DEFAULT_SEGMENTATIONS = [
    SegmentationSpec(name="natural", mode="natural", min_words=8),
    SegmentationSpec(name="sent2_stride1", mode="sentence_window", window_size=2, stride=1, min_words=5),
    SegmentationSpec(name="sent3_stride1", mode="sentence_window", window_size=3, stride=1, min_words=5),
    SegmentationSpec(name="sent2_stride2", mode="sentence_window", window_size=2, stride=2, min_words=5),
    SegmentationSpec(name="tok64", mode="token_chunk", token_size=64, min_words=5),
    SegmentationSpec(name="tok128", mode="token_chunk", token_size=128, min_words=5),
    SegmentationSpec(name="tok256", mode="token_chunk", token_size=256, min_words=5),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_natural_paragraphs(text: str, min_words: int = 5) -> List[str]:
    blocks = [normalize_text(x) for x in re.split(r"\n\s*\n", text) if normalize_text(x)]
    blocks = [b for b in blocks if len(b.split()) >= min_words]
    return blocks


def split_sentence_windows(text: str, window_size: int = 2, stride: int = 1, min_words: int = 5) -> List[str]:
    text = normalize_text(text)
    sents = [normalize_text(s) for s in _SENT_SPLIT.split(text) if normalize_text(s)]
    if not sents:
        return []
    windows: List[str] = []
    for start in range(0, len(sents), stride):
        chunk = sents[start:start + window_size]
        if len(chunk) < window_size:
            break
        joined = normalize_text(" ".join(chunk))
        if len(joined.split()) >= min_words:
            windows.append(joined)
    return windows


def split_token_chunks(text: str, token_size: int = 128, min_words: int = 5) -> List[str]:
    tokens = normalize_text(text).split()
    if not tokens:
        return []
    chunks: List[str] = []
    for start in range(0, len(tokens), token_size):
        chunk = tokens[start:start + token_size]
        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))
    return chunks


def segment_text(text: str, spec: SegmentationSpec) -> List[str]:
    if spec.mode == "natural":
        segments = split_natural_paragraphs(text, min_words=spec.min_words)
        if segments:
            return segments
        return split_sentence_windows(text, window_size=2, stride=1, min_words=max(3, spec.min_words - 2)) or [normalize_text(text)]
    if spec.mode == "sentence_window":
        return split_sentence_windows(text, window_size=spec.window_size or 2, stride=spec.stride or 1, min_words=spec.min_words) or [normalize_text(text)]
    if spec.mode == "token_chunk":
        return split_token_chunks(text, token_size=spec.token_size or 128, min_words=spec.min_words) or [normalize_text(text)]
    raise ValueError(f"Unsupported segmentation mode: {spec.mode}")
