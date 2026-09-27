"""Tests of the GPU training code in pipeline/, run on a CPU, and checks that the numpy core in
alr_ira/ makes the same draws and labels as the torch code. Needs the train extra."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from torch import nn  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import alr_ira  # noqa: E402
from alr.metrics import geometric_mean, summarize  # noqa: E402
from alr.tree import build_budget_tree, make_tree_mask  # noqa: E402
from evaluate_text import reference_rows  # noqa: E402
from specforge.core.dflash import OnlineDFlashModel, create_ira_sdpa_mask  # noqa: E402
from specforge.core.dflash_targetkd import soft_survival_gate, targetkd_loss  # noqa: E402
from specforge.modeling.target.dflash_target_model import HFDFlashTargetModel  # noqa: E402
from summarize import aggregate  # noqa: E402

BS = 16


def bare_model(num_anchors=9, seed=0, stop_ids=(), steps=0):
    model = OnlineDFlashModel.__new__(OnlineDFlashModel)
    nn.Module.__init__(model)
    model.block_size, model.num_anchors = BS, num_anchors
    model._ira_gen = model._ira_gen_device = None
    model._ira_seed = 9012 + seed
    model.kd_rollout_stop_ids = tuple(stop_ids)
    model.kd_rollout_steps = steps
    return model


def ragged_mask():
    mask = torch.ones(3, 60)
    mask[1, 40:] = 0
    mask[2, :5] = 0
    return mask


def torch_draws(mask, n2, seed=0, sampler_seed=5):
    """The uniforms and offsets the torch samplers draw, reproduced from the same seeds."""
    torch.manual_seed(sampler_seed)
    order = torch.rand(mask.shape[0], mask.shape[1] - BS + 1)
    g = torch.Generator().manual_seed(9012 + seed)
    return order.numpy(), torch.randint(2, BS - 1, (mask.shape[0], n2), generator=g).numpy()


# pipeline tests

def test_shared_ar_reference_rejects_changed_settings():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.json"
        path.write_text(json.dumps({
            "config": {"target": "target", "max_new": 256, "sample_seed": 0},
            "cells": {"task/T0": {"rows": [{"id": "p0", "ar": {"decode_ms_token": 4}}]}},
        }))
        assert reference_rows(path, "target", 256, 0)["task/T0"]["p0"]["decode_ms_token"] == 4
        with pytest.raises(ValueError):
            reference_rows(path, "different_target", 256, 0)
        with pytest.raises(ValueError):
            reference_rows(path, "target", 128, 0)


def test_tail_bucket_and_gradient():
    torch.manual_seed(4)
    logits = torch.randn(3, 19, requires_grad=True)
    target = torch.randn(3, 19).softmax(-1)
    probs, ids = target.topk(8, dim=-1)
    tail = 1 - probs.sum(-1)
    weight = torch.tensor([1.0, 0.2, 0.0])
    actual = targetkd_loss(logits, ids, probs, tail, weight)
    q_top = logits.softmax(-1).gather(1, ids)
    expected = (weight * (-(probs * q_top.log()).sum(-1) - tail * (1 - q_top.sum(-1)).log())).sum() / (
        weight.sum() + 1e-6)
    torch.testing.assert_close(actual, expected)
    (grad,) = torch.autograd.grad(actual, logits, retain_graph=True)
    (reference,) = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(grad, reference)
    assert torch.equal(grad[2], torch.zeros_like(grad[2]))


def test_ira_budget_and_offsets():
    sampled = bare_model()._sample_inrollout_anchors(48, torch.ones(2, 48), torch.device("cpu"))
    assert int(sampled["keep1"].sum() + sampled["keep2"].sum()) == sampled["alr_blocks"]
    assert ((sampled["j"] >= 2) & (sampled["j"] <= 14)).all()
    assert (sampled["parent"] < sampled["primary"].shape[1]).all()


def test_ira_attention_isolates_branches():
    mask = create_ira_sdpa_mask(torch.tensor([[3, 5]]), torch.tensor([[8, 10]]), torch.tensor([[10, 13]]),
                                torch.ones(1, 2, dtype=torch.bool), 8, 13, 4, torch.device("cpu"))[0, 0]
    assert mask[:4, :3].all() and not mask[:4, 3:8].any()
    assert mask[:4, 8:10].all() and not mask[:4, 10:13].any()
    assert mask[:4, 13:17].all() and not mask[:4, 17:].any()


def tiny_teacher(device="cpu"):
    torch.manual_seed(11)
    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().to(device)


@torch.inference_mode()
def check_rollout_matches_full_teacher_forwards(device):
    teacher = tiny_teacher(device)
    target = HFDFlashTargetModel(teacher)
    layers = [0, 1]
    target.set_capture_layers(layers)
    ids = torch.randint(0, 40, (1, 24), device=device)
    mask = torch.ones_like(ids)
    anchors = torch.tensor([[5, 12]], device=device)
    rollout = target.generate_dflash_data(ids, mask, mask, return_final_hidden=True,
                                          anchor_rollout=(anchors, 16), rollout_context_steps=13)
    for i, anchor in enumerate(anchors[0].tolist()):
        continuation = rollout.rollout_tokens[0, i, 1:15]
        sequence = torch.cat([ids[:, : anchor + 1], continuation[None]], dim=1)
        full = teacher(sequence, output_hidden_states=True, use_cache=False)
        reference = target._targetkd_final_hidden(full)[:, anchor: anchor + 15]
        torch.testing.assert_close(reference, rollout.rollout_hidden[:, i, 1:16], atol=2e-5, rtol=2e-5)
        features = torch.cat([full.hidden_states[k + 1] for k in layers], dim=-1)
        torch.testing.assert_close(features[:, anchor + 1: anchor + 14], rollout.rollout_context[:, i],
                                   atol=2e-5, rtol=2e-5)
        assert torch.equal(full.logits[:, anchor: anchor + 15].argmax(-1), rollout.rollout_tokens[:, i, 1:16])


def test_rollout_matches_full_teacher_forwards():
    check_rollout_matches_full_teacher_forwards("cpu")


@pytest.mark.gpu
def test_rollout_matches_full_teacher_forwards_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    check_rollout_matches_full_teacher_forwards("cuda")


def test_tree_is_prefix_closed():
    torch.manual_seed(7)
    nodes = build_budget_tree(torch.randn(5, 20), budget=12, branch_k=3)
    assert len(nodes) == 12
    mask = make_tree_mask(4, nodes, "cpu", torch.float32)[0, 0]
    for i, node in enumerate(nodes):
        assert node.parent < i
        visible = {0, i + 1}
        parent = node.parent
        while parent >= 0:
            visible.add(parent + 1)
            parent = nodes[parent].parent
        assert set((mask[i + 1, 4:] == 0).nonzero().flatten().tolist()) == visible


def test_pool_rounds_before_aggregating_tasks():
    rows = [{"ar": {"decode_ms_token": 4}, "draft": {"tau": 2, "rounds": 9, "decode_ms_token": 2}},
            {"ar": {"decode_ms_token": 6}, "draft": {"tau": 4, "rounds": 1, "decode_ms_token": 3}}]
    assert summarize(rows)["tau"] == pytest.approx(2.2)
    assert summarize(rows)["speedup"] == pytest.approx(2)
    assert geometric_mean([2, 8]) == pytest.approx(4)
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for i, pair in enumerate(([2, 8], [4, 4])):
            path = Path(directory) / f"{i}.json"
            path.write_text(json.dumps({"cells": {f"{task}/T0": {"summary": {"tau": tau, "speedup": 2}}
                                                  for task, tau in zip(["a", "b"], pair)}}))
            paths.append(path)
        assert aggregate(paths)["T0"]["tau"]["mean"] == pytest.approx(4)


# the numpy core against the torch code

@pytest.mark.parametrize("num_anchors", [9, 128])
def test_alr_anchor_draw_matches_torch(num_anchors):
    mask = ragged_mask()
    torch.manual_seed(5)
    anchors, keep = bare_model(num_anchors)._sample_anchor_positions(60, mask, torch.device("cpu"))
    order, _ = torch_draws(mask, 1)
    a, k = alr_ira.sample_anchors(mask.numpy(), BS, num_anchors, order=order)
    assert np.array_equal(a, anchors.numpy()) and np.array_equal(k, keep.numpy())


@pytest.mark.parametrize("num_anchors, seed", [(9, 0), (128, 2)])
def test_ira_split_matches_torch(num_anchors, seed):
    mask = ragged_mask()
    torch.manual_seed(5)
    t = bare_model(num_anchors, seed)._sample_inrollout_anchors(60, mask, torch.device("cpu"))
    order, offsets = torch_draws(mask, t["j"].shape[1], seed)
    s = alr_ira.sample_inrollout_anchors(mask.numpy(), BS, num_anchors, order=order, offsets=offsets)
    for key in ("primary", "keep1", "j", "keep2"):
        assert np.array_equal(s[key], t[key].numpy()), key
    kept = t["keep2"].numpy()
    assert np.array_equal(s["parent"][kept], t["parent"].numpy()[kept])
    assert s["blocks"] == t["alr_blocks"]


def test_slot_matched_control_matches_torch():
    mask = ragged_mask()
    torch.manual_seed(5)
    anchors, keep, cap = bare_model(128)._sample_slotmatched_anchors(60, mask, torch.device("cpu"))
    max_n = anchors.shape[1]
    order, offsets = torch_draws(mask, max_n // 2)
    a, k, c = alr_ira.sample_slotmatched_anchors(mask.numpy(), BS, 128, order=order, offsets=offsets)
    assert np.array_equal(a, anchors.numpy()) and np.array_equal(k, keep.numpy())
    assert np.array_equal(c, cap.numpy())


def test_gates_alive_and_depth_match_torch():
    g = torch.Generator().manual_seed(3)
    prob = torch.rand(4, 6, BS, generator=g)
    scored = torch.rand(4, 6, BS, generator=g) > 0.2
    torch.testing.assert_close(torch.from_numpy(alr_ira.soft_survival_gate(prob.numpy(), scored.numpy())).float(),
                               soft_survival_gate(prob, scored), rtol=1e-5, atol=1e-6)
    model = bare_model(stop_ids=(7, 9), steps=4)
    corpus = torch.randint(0, 12, (4, 6, BS), generator=g)
    greedy = torch.where(torch.rand(4, 6, BS, generator=g) < 0.8, corpus, torch.randint(0, 12, (4, 6, BS), generator=g))
    valid = torch.rand(4, 6, BS, generator=g) > 0.1
    hard = model._survival_gate(greedy.reshape(-1, 1), corpus, valid)
    assert np.array_equal(alr_ira.hard_survival_gate(corpus.numpy(), greedy.numpy(), valid.numpy()), hard.numpy())
    assert np.array_equal(alr_ira.rollout_alive(corpus.numpy(), (7, 9)), model._rollout_alive(corpus).numpy())
    assert np.array_equal(alr_ira.depth_mask(BS, 4), model._rollout_depth("cpu", torch.float32).numpy())


def test_kd_loss_matches_torch():
    torch.manual_seed(4)
    logits = torch.randn(12, 50, dtype=torch.float64)
    probs, ids = torch.randn(12, 50).softmax(-1).topk(8, dim=-1)
    tail = (1 - probs.sum(-1)).clamp_min(0)
    weight = torch.rand(12)
    for lam in (1.0, 1.013441):
        expected = targetkd_loss(logits.float(), ids, probs, tail, weight, lambda_scale=lam).item()
        got = alr_ira.kd_loss(logits.numpy(), ids.numpy(), probs.numpy(), tail.numpy(), weight.numpy(), lam)
        assert got == pytest.approx(expected, rel=1e-5)
