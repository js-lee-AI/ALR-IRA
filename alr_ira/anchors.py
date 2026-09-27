"""Anchor placement for ALR, ALR + IRA and the slot-matched control, in numpy.

Each function mirrors the sampler of the training code in
pipeline/specforge/core/dflash.py. `loss_mask` is [batch, seq_len]. The random
draws can be passed in (`order`, `offsets`), which the tests use to compare
against the torch code draw for draw.
"""

from __future__ import annotations

import numpy as np


def _draw(loss_mask, block_size, num_anchors, rng, order):
    loss_mask = np.asarray(loss_mask)
    bsz, seq_len = loss_mask.shape
    max_anchor = max(seq_len - block_size, 0)
    valid = loss_mask[:, : max_anchor + 1] > 0.5
    counts = valid.sum(axis=1)
    max_n = min(num_anchors, int(counts.max()) - 1)
    if max_n <= 0:
        raise ValueError("every sequence needs more supervised positions than one block")
    if order is None:
        order = np.random.default_rng(rng).random((bsz, max_anchor + 1))
    order = np.where(valid, np.asarray(order, dtype=np.float64), 2.0)
    positions = np.where(valid, np.arange(max_anchor + 1)[None, :], seq_len + 1)
    drawn = np.take_along_axis(positions, np.argsort(order, axis=1, kind="stable"), axis=1)
    m = np.minimum(counts, max_n)
    return drawn, m, max_n


def sample_anchors(loss_mask, block_size=16, num_anchors=128, rng=None, order=None):
    """ALR's anchor draw: m = min(v, max_n) random positions per sequence, sorted.

    Returns (anchors, keep), both [batch, max_n]. Dropped entries are 0 with keep False.
    """
    drawn, m, max_n = _draw(loss_mask, block_size, num_anchors, rng, order)
    anchors = np.sort(drawn[:, :max_n], axis=1)
    keep = np.arange(max_n)[None, :] < m[:, None]
    return np.where(keep, anchors, 0), keep


def sample_inrollout_anchors(loss_mask, block_size=16, num_anchors=128, rng=None, order=None, offsets=None):
    """IRA's split of the same m blocks (Section 3.2, Eq. 4).

    The first ceil(m/2) drawn positions are primary blocks, which run the target
    rollout. Secondary block i sits inside the rollout of the i-th drawn primary at
    offset j ~ U{2, ..., block_size - 2}. Returns a dict with
    primary [b, n1], keep1, parent [b, n2] (index into the sorted primaries),
    j [b, n2], keep2, and rollouts (number of primary blocks, which run a rollout)
    and blocks (m summed over the batch, the same total as ALR).
    """
    drawn, m, max_n = _draw(loss_mask, block_size, num_anchors, rng, order)
    seq_len = np.asarray(loss_mask).shape[1]
    n1, n2 = (max_n + 1) // 2, max_n // 2
    in_order = np.arange(n1)[None, :] < ((m + 1) // 2)[:, None]
    first = np.where(in_order, drawn[:, :n1], seq_len + 1)
    by_position = np.argsort(first, axis=1, kind="stable")
    primary = np.take_along_axis(first, by_position, axis=1)
    keep1 = np.take_along_axis(in_order, by_position, axis=1)
    primary = np.where(keep1, primary, 0)
    parent = np.argsort(by_position, axis=1, kind="stable")[:, :n2]
    keep2 = np.arange(n2)[None, :] < (m // 2)[:, None]
    if offsets is None:
        offsets = np.random.default_rng(rng).integers(2, block_size - 1, size=(len(m), n2))
    return dict(primary=primary, keep1=keep1, parent=parent, j=np.asarray(offsets), keep2=keep2,
                rollouts=int(keep1.sum()), blocks=int(m.sum()))


def sample_slotmatched_anchors(loss_mask, block_size=16, num_anchors=128, rng=None, order=None, offsets=None):
    """The slot-matched control: ALR's m corpus anchors, with the blocks IRA would move
    into rollouts capped at K - j supervised slots.

    Returns (anchors, keep, slot_cap), each [batch, max_n]. Slot k is trained when k <= slot_cap.
    """
    drawn, m, max_n = _draw(loss_mask, block_size, num_anchors, rng, order)
    bs = block_size
    n2 = max_n // 2
    if offsets is None:
        offsets = np.random.default_rng(rng).integers(2, bs - 1, size=(len(m), n2))
    offsets = np.asarray(offsets)
    full = np.full((len(m), max_n), bs - 1)
    if n2 > 0:
        i = np.arange(max_n)[None, :] - ((m + 1) // 2)[:, None]
        capped = (i >= 0) & (i < (m // 2)[:, None])
        j_at = np.take_along_axis(offsets, np.clip(i, 0, n2 - 1), axis=1)
        cap_order = np.where(capped, bs - 1 - j_at, full)
    else:
        cap_order = full
    by_position = np.argsort(drawn[:, :max_n], axis=1, kind="stable")
    anchors = np.take_along_axis(drawn[:, :max_n], by_position, axis=1)
    slot_cap = np.take_along_axis(cap_order, by_position, axis=1)
    keep = np.arange(max_n)[None, :] < m[:, None]
    return np.where(keep, anchors, 0), keep, np.where(keep, slot_cap, full)
