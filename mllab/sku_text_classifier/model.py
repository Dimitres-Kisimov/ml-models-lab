"""fastText-style classifier: hashed char n-grams -> EmbeddingBag(mean) -> softmax."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mllab.sku_text_classifier.hashing import hash_document


def build_bag(texts: list[str], n_buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten documents into (indices, offsets) for nn.EmbeddingBag."""
    flat: list[int] = []
    offsets: list[int] = []
    pos = 0
    for t in texts:
        offsets.append(pos)
        ids = hash_document(t, n_buckets)
        flat.extend(ids)
        pos += len(ids)
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


class HashedBagClassifier(nn.Module):
    def __init__(self, n_buckets: int, n_classes: int, dim: int = 24):
        super().__init__()
        self.embed = nn.EmbeddingBag(n_buckets, dim, mode="mean")
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, indices, offsets):
        return self.fc(self.embed(indices, offsets))


def inverse_frequency_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_classifier(
    texts: list[str],
    labels: np.ndarray,
    n_classes: int,
    n_buckets: int = 2**18,
    dim: int = 24,
    epochs: int = 60,
    lr: float = 0.5,
    seed: int = 0,
) -> HashedBagClassifier:
    torch.manual_seed(seed)
    model = HashedBagClassifier(n_buckets, n_classes, dim=dim)
    indices, offsets = build_bag(texts, n_buckets)
    y = torch.tensor(labels, dtype=torch.long)
    weights = inverse_frequency_weights(labels, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(indices, offsets)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict(model: HashedBagClassifier, texts: list[str], n_buckets: int) -> np.ndarray:
    indices, offsets = build_bag(texts, n_buckets)
    with torch.no_grad():
        logits = model(indices, offsets)
        return logits.argmax(dim=1).numpy()
