"""The distillation loss of Eq. 1 in numpy: soft cross-entropy against the target's
top-k probabilities plus one tail bucket, as a weighted mean over slots."""

from __future__ import annotations

import numpy as np

TAIL_SUM_CLAMP = 1.0 - 1e-6


def topk_labels(target_probs, k: int = 8):
    """Top-k token ids and probabilities of each label distribution, and the remaining mass.

    `target_probs` is [..., vocab]. Ties go to the lower token id.
    """
    p = np.asarray(target_probs, dtype=np.float64)
    idx = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    prob = np.take_along_axis(p, idx, axis=-1)
    tail = np.clip(1.0 - prob.sum(axis=-1), 0.0, None)
    return idx, prob, tail


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kd_loss(draft_logits, topk_idx, topk_prob, tail_mass, weight, lambda_scale: float = 1.0,
            tail_bucket: bool = True) -> float:
    """lambda * sum_s w_s l_s / (sum_s w_s + 1e-6) over flattened slots s.

    l_s = -sum_i p_i log q_i - tail * log(1 - sum_i q_i), with i over the top-k ids.
    Shapes are [S, vocab], [S, k], [S, k], [S] and [S].
    """
    log_q = log_softmax(draft_logits)
    log_q_top = np.take_along_axis(log_q, np.asarray(topk_idx), axis=-1)
    kd = -(np.asarray(topk_prob) * log_q_top).sum(axis=-1)
    if tail_bucket:
        q_sum = np.minimum(np.exp(log_q_top).sum(axis=-1), TAIL_SUM_CLAMP)
        kd = kd - np.asarray(tail_mass) * np.log1p(-q_sum)
    w = np.asarray(weight, dtype=np.float64)
    return float(lambda_scale * (w * kd).sum() / (w.sum() + 1e-6))
