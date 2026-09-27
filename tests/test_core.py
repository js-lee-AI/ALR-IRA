import math

import numpy as np
import pytest

import alr_ira
from alr_ira.toy import make_world

BS = 16
K = BS - 1


def markov_next_probs(world):
    return lambda tokens: world.target[tokens[-2], tokens[-1]]


def test_envelope_of_eq1():
    w = alr_ira.slot_envelope(BS, gamma=2.0)
    assert w[1] == 1.0 and w[2] == pytest.approx(math.exp(-0.5)) and w[15] == pytest.approx(math.exp(-7))


def test_soft_gate_is_the_product_of_earlier_slot_probabilities():
    p = np.array([0.3, 0.9, 0.5, 0.8, 0.2])
    g = alr_ira.soft_survival_gate(p, np.array([0, 1, 1, 1, 1]))
    assert g.tolist() == pytest.approx([1.0, 1.0, 0.9, 0.45, 0.36])


def test_hard_gate_stops_after_the_first_mismatch():
    g = alr_ira.hard_survival_gate([7, 1, 2, 9, 4], [0, 1, 2, 3, 4])
    assert g.tolist() == [1, 1, 1, 1, 0]


def test_alr_labels_follow_the_greedy_rollout():
    world = make_world(seed=3)
    f = markov_next_probs(world)
    prefix = [int(t) for t in world.corpus[0, :5]]
    rollout, labels = alr_ira.alr_labels(f, prefix, BS)
    assert len(rollout) == K and labels.shape == (K, world.target.shape[0])
    ctx = list(prefix)
    for k in range(K):
        assert np.allclose(labels[k], f(ctx)) and rollout[k] == int(np.argmax(f(ctx)))
        ctx.append(rollout[k])
    # slot 1 gets the same label under KD, erase and ALR (Section 3.2)
    kd = alr_ira.corpus_labels(f, world.corpus[0], anchor=4, block_size=BS)
    assert np.allclose(kd[0], labels[0])
    # on the greedy rollout the hard gate keeps every slot
    assert alr_ira.hard_survival_gate([prefix[-1]] + rollout, [prefix[-1]] + rollout).min() == 1


def test_ira_keeps_the_block_count_and_halves_the_rollouts():
    mask = np.ones((4, 300))
    mask[1, 200:] = 0
    s = alr_ira.sample_inrollout_anchors(mask, BS, num_anchors=128, rng=0)
    _, keep = alr_ira.sample_anchors(mask, BS, num_anchors=128, rng=0)
    m = keep.sum(axis=1)
    assert s["blocks"] == keep.sum()
    assert s["rollouts"] == int(np.ceil(m / 2).sum())
    assert (s["keep2"].sum(axis=1) == m // 2).all()
    assert ((s["j"] >= 2) & (s["j"] <= K - 1)).all()
    # every kept secondary block points at a kept primary
    rows, cols = np.nonzero(s["keep2"])
    assert s["keep1"][rows, s["parent"][rows, cols]].all()


def test_secondary_blocks_supervise_seven_slots_on_average():
    # j ~ U{2, ..., K - 1} leaves K - j supervised slots, whose mean is 7 for K = 15
    assert np.mean([K - j for j in range(2, K)]) == 7
    weight = np.ones((1, BS))
    weight[:, 0] = 0
    tokens = np.arange(BS)[None, :]
    for j in range(2, K):
        w2, t2, _ = alr_ira.secondary_blocks(weight, tokens, np.array([0]), np.array([j]), np.array([True]))
        assert w2.sum() == K - j and t2[0, 0] == j


def test_secondary_blocks_match_eq4_elementwise():
    rng = np.random.default_rng(0)
    n1, n2 = 5, 4
    weight = rng.random((n1, BS))
    tokens = rng.integers(0, 50, size=(n1, BS))
    alive = (rng.random((n1, BS)) > 0.2).astype(float)
    parent = rng.integers(0, n1, n2)
    offset = rng.integers(2, K, n2)
    keep = np.array([True, True, False, True])
    w2, t2, a2 = alr_ira.secondary_blocks(weight, tokens, parent, offset, keep, alive)
    for i in range(n2):
        for k in range(BS):
            inside = offset[i] + k <= K
            src = min(offset[i] + k, K)
            assert w2[i, k] == (weight[parent[i], src] if inside and keep[i] and k > 0 else 0)
            assert t2[i, k] == (tokens[parent[i], src] if inside else -1)
            assert a2[i, k] == (alive[parent[i], src] if inside else 0)


def test_slot_matched_control_caps_the_displaced_blocks():
    # the blocks IRA would move keep their corpus anchor but train only K - j slots
    mask = np.ones((3, 200))
    offsets = np.random.default_rng(1).integers(2, K, size=(3, 64))
    _, keep, cap = alr_ira.sample_slotmatched_anchors(mask, BS, 128, rng=1, offsets=offsets)
    for b in range(3):
        m = int(keep[b].sum())
        expected = [K] * ((m + 1) // 2) + list(K - offsets[b, : m // 2])
        assert sorted(cap[b][keep[b]]) == sorted(expected)


def test_rollout_alive_and_depth():
    # slots up to and including the first end-of-turn token, and an anchor that is one does not count
    assert alr_ira.rollout_alive([5, 9, 3, 9, 1], stop_ids={3}).tolist() == [1, 1, 1, 0, 0]
    assert alr_ira.rollout_alive([3, 9, 9, 3, 1], stop_ids={3}).tolist() == [1, 1, 1, 1, 0]
    assert alr_ira.depth_mask(BS, 4).tolist() == [1] * 6 + [0] * 10
    assert alr_ira.depth_mask(BS, 14).min() == 1


def test_kd_loss_is_smallest_at_the_label():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(20) * 0.3, size=6)
    idx, prob, tail = alr_ira.topk_labels(p, k=8)
    w = np.ones(6)
    at_label = alr_ira.kd_loss(np.log(p), idx, prob, tail, w)
    for _ in range(5):
        assert at_label < alr_ira.kd_loss(np.log(p) + rng.normal(size=p.shape), idx, prob, tail, w)


def test_metrics():
    assert alr_ira.mat([10, 20], [4, 6]) == 3.0
    assert alr_ira.geomean([2, 8]) == pytest.approx(4)
    mean, sd = alr_ira.aggregate([[2, 8], [4, 4]])
    assert mean == pytest.approx(4) and sd == 0
    assert alr_ira.speedup([30, 10], [10, 10]) == 2
    assert alr_ira.gain(3.84, 3.54) == pytest.approx(8.4745, abs=1e-4)
