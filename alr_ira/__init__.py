"""ALR and IRA: training labels and anchors for block drafters on a fixed corpus.

    import alr_ira
    rollout, labels = alr_ira.alr_labels(next_probs, prefix, block_size=16)
    split = alr_ira.sample_inrollout_anchors(loss_mask, block_size=16, num_anchors=128, rng=0)

The label, slot and anchor logic of the paper runs on numpy alone, so this package
imports without torch. The GPU training and evaluation code is in pipeline/.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .anchors import sample_anchors, sample_inrollout_anchors, sample_slotmatched_anchors
from .labels import (alr_labels, corpus_labels, depth_mask, greedy_rollout, hard_survival_gate,
                     rollout_alive, secondary_blocks, slot_envelope, soft_survival_gate)
from .loss import kd_loss, topk_labels
from .metrics import aggregate, gain, geomean, mat, speedup
from .toy import format_results, make_world, run_toy

__all__ = [
    "__version__",
    "alr_labels", "corpus_labels", "greedy_rollout", "slot_envelope", "soft_survival_gate",
    "hard_survival_gate", "rollout_alive", "depth_mask", "secondary_blocks",
    "sample_anchors", "sample_inrollout_anchors", "sample_slotmatched_anchors",
    "topk_labels", "kd_loss",
    "mat", "geomean", "aggregate", "speedup", "gain",
    "run_toy", "make_world", "format_results",
]
