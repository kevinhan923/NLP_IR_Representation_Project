import re
from typing import List, Dict


class StructuralSegmenter:
    """
    Split on double-newlines (natural paragraph breaks in the raw text).
    Best for documents that already have editorial paragraph structure (Wikipedia, news).
    Avoids artificial boundaries from fixed token counts.
    """

    def __init__(self, min_chars: int = 30):
        self.min_chars = min_chars

    def segment(self, text: str) -> List[Dict]:
        raw = re.split(r"\n\n+", text.strip())
        paragraphs = []
        for i, chunk in enumerate(raw):
            chunk = chunk.strip()
            if len(chunk) >= self.min_chars:
                paragraphs.append({"text": chunk, "position": i, "length": len(chunk.split())})
        return paragraphs if paragraphs else [{"text": text, "position": 0, "length": len(text.split())}]


class SentenceWindowSegmenter:
    """
    Sliding window over sentences. Fallback when text has no paragraph breaks.
    Principle: a window of W sentences with stride S creates overlapping passages
    so that relevant sentences aren't split across boundaries.
    """

    def __init__(self, window: int = 3, stride: int = 1):
        self.window = window
        self.stride = stride

    def segment(self, text: str) -> List[Dict]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= self.window:
            return [{"text": text, "position": 0, "length": len(text.split())}]
        paragraphs = []
        i = 0
        while i + self.window <= len(sentences):
            chunk = " ".join(sentences[i:i + self.window])
            paragraphs.append({"text": chunk, "position": i, "length": len(chunk.split())})
            i += self.stride
        return paragraphs
