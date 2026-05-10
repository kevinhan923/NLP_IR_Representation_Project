import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict


class MaxAggregator:
    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        return float(scores.max())


class TopKMeanAggregator:
    def __init__(self, k: int = 3):
        self.k = k

    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        k = min(self.k, len(scores))
        return float(np.sort(scores)[::-1][:k].mean())


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LearnedMLPAggregator:
    """
    4-feature MLP aggregator.
    Input per paragraph: [para_score, norm_position, norm_length, query_term_overlap]
    Aggregated doc score = max of MLP outputs over all paragraphs.
    Training: binary cross-entropy on MS MARCO passage relevance labels.

    Note: norm_position uses the loop index (not the segment's position field) so that
    non-contiguous position values from StructuralSegmenter do not exceed 1.0.
    """

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.model = _MLP()

    def _make_features(self, scores: np.ndarray, features: List[Dict]) -> torch.Tensor:
        n = len(scores)
        rows = []
        for i, (score, feat) in enumerate(zip(scores, features)):
            # Use loop index for normalization to handle non-contiguous position values
            norm_pos = i / max(n - 1, 1)
            norm_len = min(feat.get("length", 100) / self.max_length, 1.0)
            overlap = feat.get("overlap", 0.0)
            rows.append([float(score), norm_pos, norm_len, overlap])
        return torch.tensor(rows, dtype=torch.float32)

    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        x = self._make_features(scores, features)
        with torch.no_grad():
            outputs = self.model(x)
        return float(outputs.max().item())

    def train_on_examples(self, examples: List[Dict]):
        """
        Train the MLP on labeled (scores, features, label) tuples.
        examples: list of {"scores": np.array, "features": [...], "label": 0|1}
        """
        if not examples:
            return
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        self.model.train()
        for epoch in range(30):
            total_loss = 0.0
            for ex in examples:
                x = self._make_features(ex["scores"], ex["features"])
                pred = self.model(x).max()
                label = torch.tensor(float(ex["label"]))
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        self.model.eval()
