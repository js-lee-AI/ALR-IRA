"""Slot weights and labels of one draft block (paper Section 3), in numpy.

A block has `block_size` slots. Slot 0 holds the anchor token and slots 1 .. K
(K = block_size - 1) are predicted. Arrays are indexed [..., slot].
"""

from __future__ import annotations

import numpy as np


def slot_envelope(block_size: int = 16, gamma: float | None = 2.0) -> np.ndarray:
    """w_k = exp(-(k - 1) / gamma) for every slot (Eq. 1). Slot 0 gets 1 and is masked elsewhere."""
    k = np.arange(block_size, dtype=np.float64)
    if gamma is None or gamma <= 0:
        return np.ones(block_size)
    return np.exp(-np.clip(k - 1, 0, None) / gamma)


def soft_survival_gate(label_prob, scored) -> np.ndarray:
    """Erase gate of Eq. 2, g_k = prod_{1 <= j < k} p_T(x_{a+j} | x_{<a+j}).

    `label_prob[..., j]` is the target's probability of the token at slot j and
    `scored[..., j]` says whether slot j votes. Slot 0 and unscored slots count as 1.
    """
    p = np.asarray(label_prob, dtype=np.float64)
    c = np.where(np.asarray(scored) > 0, p, 1.0)
    shifted = np.concatenate([np.ones_like(c[..., :1]), c[..., :-1]], axis=-1)
    return np.cumprod(shifted, axis=-1)


def hard_survival_gate(corpus_tokens, greedy_tokens, valid=None) -> np.ndarray:
    """The hard variant of Eq. 2, g_k = prod_{j < k} 1[x_{a+j} = y_hat_{a+j}]."""
    corpus_tokens = np.asarray(corpus_tokens)
    agree = corpus_tokens == np.asarray(greedy_tokens)
    if valid is not None:
        agree |= ~np.asarray(valid, dtype=bool)
    agree[..., 0] = True
    surv = np.cumprod(agree.astype(np.float64), axis=-1)
    gate = np.ones_like(surv)
    gate[..., 1:] = surv[..., :-1]
    return gate


def rollout_alive(tokens, stop_ids) -> np.ndarray:
    """1 for slots up to and including the first end-of-turn token of a rollout, 0 after it."""
    tokens = np.asarray(tokens)
    stop = np.isin(tokens, list(stop_ids))
    stop[..., 0] = False
    cont = np.cumprod(~stop, axis=-1).astype(np.float64)
    alive = np.ones_like(cont)
    alive[..., 1:] = cont[..., :-1]
    return alive


def depth_mask(block_size: int, rollout_depth: int | None) -> np.ndarray:
    """Slots a rollout of depth R labels (0 .. R + 1). Full depth is R = block_size - 2."""
    k = np.arange(block_size)
    if not rollout_depth or rollout_depth >= block_size - 2:
        return np.ones(block_size)
    return (k <= rollout_depth + 1).astype(np.float64)


def greedy_rollout(next_probs, prefix, steps: int) -> list:
    """Continue `prefix` greedily for `steps` tokens. `next_probs(tokens)` returns p_T(. | tokens).

    Ties go to the lower token id, as in the training code (argmax).
    """
    out = list(prefix)
    for _ in range(steps):
        out.append(int(np.argmax(next_probs(out))))
    return out[len(prefix):]


def alr_labels(next_probs, prefix, block_size: int = 16):
    """ALR labels of one block (Eq. 3).

    Returns the rollout y*_1 .. y*_{K} and the label pi_k = p_T(. | x_{<=a}, y*_{<k})
    for every slot k = 1 .. K, stacked as an array [K, vocab].
    """
    K = block_size - 1
    rollout = greedy_rollout(next_probs, prefix, K)
    labels = [np.asarray(next_probs(list(prefix) + rollout[: k - 1]), dtype=np.float64) for k in range(1, K + 1)]
    return rollout, np.stack(labels)


def corpus_labels(next_probs, sequence, anchor: int, block_size: int = 16):
    """KD labels along the corpus (Section 3.1): pi_k = p_T(. | x_{<a+k}) for k = 1 .. K."""
    K = block_size - 1
    seq = list(sequence)
    return np.stack([np.asarray(next_probs(seq[: anchor + k]), dtype=np.float64) for k in range(1, K + 1)])


def secondary_blocks(weight, rollout_tokens, parent, offset, keep, alive=None):
    """Labels of IRA's secondary blocks (Eq. 4).

    `weight` [n1, block_size] and `rollout_tokens` [n1, block_size] belong to the
    primary blocks, slot 0 being the anchor. `weight` is the primaries' slot mask
    before the envelope, which the training code applies afterwards at each block's
    own slot index. Secondary block i sits inside primary `parent[i]` at offset
    j = `offset[i]`, takes y*_j as its anchor and inherits the primary's label and
    mask at slot j + k. Slots past the block end get weight 0 and token -1.
    Returns (weight2, tokens2, alive2), each [n2, block_size].
    """
    weight = np.asarray(weight, dtype=np.float64)
    rollout_tokens = np.asarray(rollout_tokens)
    parent = np.asarray(parent)
    offset = np.asarray(offset)
    keep = np.asarray(keep, dtype=bool)
    bs = weight.shape[-1]
    k = np.arange(bs)
    src = offset[:, None] + k[None, :]
    inside = src <= bs - 1
    src = np.minimum(src, bs - 1)
    rows = parent[:, None]
    weight2 = weight[rows, src] * (inside & keep[:, None] & (k[None, :] > 0))
    tokens2 = np.where(inside, rollout_tokens[rows, src], -1)
    alive2 = None
    if alive is not None:
        alive2 = np.asarray(alive, dtype=np.float64)[rows, src] * inside
    return weight2, tokens2, alive2
