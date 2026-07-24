"""From-scratch evaluation metrics (no scikit-learn).

Everything here is numpy-only so the numpy models stay CI-testable without torch.
"""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) formulation, ties averaged."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    n = len(score)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under precision-recall via the AP sum)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    # AP = sum over thresholds of (recall_i - recall_{i-1}) * precision_i
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    k = min(k, len(score))
    top = np.argsort(-score, kind="mergesort")[:k]
    return float(y_true[top].mean()) if k > 0 else float("nan")


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(bin_conf, bin_acc, bin_count)`` for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        mask = idx == b
        cnt[b] = mask.sum()
        if mask.any():
            conf[b] = prob[mask].mean()
            acc[b] = y_true[mask].mean()
    return conf, acc, cnt


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[t, p] += 1
    return cm


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    """Mean absolute scaled error, scaled by in-sample seasonal-naive MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        m = 1
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    denom = max(denom, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    """Root mean squared scaled error (M5 metric)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        m = 1
    denom = np.mean((y_train[m:] - y_train[:-m]) ** 2)
    denom = max(denom, 1e-8)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / denom))
