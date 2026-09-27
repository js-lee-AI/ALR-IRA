"""A toy version of the paper's setting that runs on a CPU in under a second.

The target is an order-2 Markov chain over a small vocabulary whose next token depends
mostly on the last token and less on the one before. The corpus is written by a mixture
of the target and a foreign chain, so its continuations often leave the target's greedy
path. The drafter is a table with one row per anchor token and slot, so it cannot see the
token before the anchor. That token comes from the corpus writer in training and from the
target at decoding time, which is the context gap that IRA closes. The drafter is fit in
closed form as the weighted mean of its labels, the minimiser of Eq. 1 with full label
distributions in place of the top 8 and the tail. Accepted length comes from greedy chain
verification against the target.

Every objective sees the same corpus and trains the same number of blocks. Only the
labels, the slot weights and, for IRA, the anchor positions differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchors import sample_anchors, sample_inrollout_anchors
from .labels import secondary_blocks, slot_envelope, soft_survival_gate

METHODS = ("dflash", "kd", "erase", "alr", "alr_ira")
LABELS = {"dflash": "DFlash", "kd": "KD", "erase": "Erase", "alr": "ALR", "alr_ira": "ALR + IRA"}


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class ToyWorld:
    target: np.ndarray      # [V, V, V], p_T(x_t | x_{t-2}, x_{t-1})
    writer: np.ndarray      # [V, V, V], the chain that wrote the corpus
    corpus: np.ndarray      # [n, length] token ids
    prompts: np.ndarray     # [n_prompts, 2] held-out openings from the writer

    @property
    def greedy(self):
        return self.target.argmax(axis=-1)


def _sample(chain, n, length, rng):
    V = chain.shape[0]
    x = np.empty((n, length), dtype=np.int64)
    x[:, :2] = rng.integers(0, V, size=(n, 2))
    for t in range(2, length):
        cum = chain[x[:, t - 2], x[:, t - 1]].cumsum(axis=-1)
        x[:, t] = np.minimum((rng.random(n)[:, None] > cum).sum(axis=-1), V - 1)
    return x


def _chain(vocab, last, before, rng):
    # logits = last * A[x_{t-1}] + before * B[x_{t-2}, x_{t-1}]
    a = rng.normal(size=(1, vocab, vocab))
    b = rng.normal(size=(vocab, vocab, vocab))
    return _softmax(last * a + before * b)


def make_world(vocab=24, n_seq=400, length=48, n_prompts=200, off_policy=0.5, last=4.0, before=2.0, seed=0):
    """A target, a corpus written by a mixture of the target and a foreign chain, and held-out prompts.

    `off_policy` is the weight of the foreign chain in the writer's mixture.
    """
    rng = np.random.default_rng(seed)
    target = _chain(vocab, last, before, rng)
    foreign = _chain(vocab, last, before, rng)
    writer = (1 - off_policy) * target + off_policy * foreign
    corpus = _sample(writer, n_seq, length, rng)
    prompts = _sample(writer, n_prompts, 2, rng)
    return ToyWorld(target, writer, corpus, prompts)


def _rollouts(world, prev, anchor, steps):
    """Greedy rollouts from (x_{a-1}, x_a). Returns tokens [n, steps + 2] with the anchor
    in column 0, and the label of slot k (k = 1 .. steps + 1) in labels[:, k]."""
    P, G = world.target, world.greedy
    n = len(anchor)
    tokens = np.empty((n, steps + 2), dtype=np.int64)
    labels = np.zeros((n, steps + 2, P.shape[0]))
    tokens[:, 0] = anchor
    p2, p1 = prev, anchor
    for k in range(1, steps + 2):
        labels[:, k] = P[p2, p1]
        tokens[:, k] = G[p2, p1]
        p2, p1 = p1, tokens[:, k]
    return tokens, labels


def training_blocks(world, method, block_size=16, num_anchors=8, gamma=2.0, seed=0):
    """Blocks of one objective, as (anchor token, labels [n, bs, V], weights [n, bs]) plus a count of rollouts."""
    x = world.corpus
    P = world.target
    n_seq, L = x.shape
    bs = block_size
    mask = np.ones_like(x, dtype=np.float64)
    mask[:, 0] = 0      # the order-2 target needs one token before the anchor
    env = slot_envelope(bs, gamma)
    rng = np.random.default_rng(seed)
    if method == "alr_ira":
        s = sample_inrollout_anchors(mask, bs, num_anchors, rng=rng)
        rows, cols = np.nonzero(s["keep1"])
        a = s["primary"][rows, cols]
    else:
        anchors, keep = sample_anchors(mask, bs, num_anchors, rng=rng)
        rows, cols = np.nonzero(keep)
        a = anchors[rows, cols]
    prev, anchor = x[rows, a - 1], x[rows, a]
    valid = np.ones((len(a), bs))
    valid[:, 0] = 0
    weights = valid * env

    if method in ("dflash", "kd", "erase"):
        pos = a[:, None] + np.arange(bs)[None, :]
        cont = x[rows[:, None], pos]                    # corpus tokens of the block, slot 0 the anchor
        before = x[rows[:, None], np.maximum(pos - 1, 0)]
        before2 = x[rows[:, None], np.maximum(pos - 2, 0)]
        if method == "dflash":
            labels = np.eye(P.shape[0])[cont]
        else:
            labels = P[before2, before]                 # p_T(. | x_{<a+k}) along the corpus
        labels[:, 0] = 0
        if method == "erase":
            prob = np.take_along_axis(labels, cont[..., None], axis=-1)[..., 0]
            weights = weights * soft_survival_gate(prob, np.arange(bs)[None, :] > 0)
        return anchor, labels, weights, 0

    tokens, labels = _rollouts(world, prev, anchor, bs - 2)
    if method == "alr":
        return anchor, labels, weights, len(a)

    # ALR + IRA: secondary block i sits inside the rollout of the primary it was paired with.
    index = np.full(s["keep1"].shape, -1)
    index[rows, cols] = np.arange(len(a))
    srow, scol = np.nonzero(s["keep2"])
    parent = index[srow, s["parent"][srow, scol]]
    j = s["j"][srow, scol]
    assert (parent >= 0).all()
    valid2, tok2, _ = secondary_blocks(valid, tokens, parent, j, np.ones(len(j), dtype=bool))
    w2 = valid2 * env                                   # w_k at the secondary block's own slot k (Eq. 4)
    src = np.minimum(j[:, None] + np.arange(bs)[None, :], bs - 1)
    lab2 = labels[parent[:, None], src] * (w2 > 0)[..., None]
    return (np.concatenate([anchor, tok2[:, 0]]), np.concatenate([labels, lab2]),
            np.concatenate([weights, w2]), len(a))


def fit_drafter(anchor, labels, weights, vocab):
    """The weighted mean label of every (slot, anchor token) cell, uniform where no block trains it."""
    bs = labels.shape[1]
    num = np.zeros((bs, vocab, vocab))
    den = np.zeros((bs, vocab))
    for k in range(1, bs):
        np.add.at(num[k], anchor, weights[:, k, None] * labels[:, k])
        np.add.at(den[k], anchor, weights[:, k])
    q = np.full_like(num, 1.0 / vocab)
    seen = den > 0
    q[seen] = num[seen] / den[seen][:, None]
    return q


def chain_decode(world, drafter, new_tokens=64):
    """Greedy chain verification. Returns (committed, rounds) per prompt."""
    G = world.greedy
    K = drafter.shape[0] - 1
    draft = drafter.argmax(axis=-1)                    # [bs, V], the drafter's pick per slot and anchor
    p2, p1 = world.prompts[:, 0].copy(), world.prompts[:, 1].copy()
    n = len(p1)
    committed = np.zeros(n, dtype=np.int64)
    rounds = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    while not done.all():
        guess = draft[1:, p1].T                        # [n, K]
        path = np.empty((n, K + 1), dtype=np.int64)
        a, b = p2, p1
        for k in range(K + 1):
            path[:, k] = G[a, b]
            a, b = b, path[:, k]
        match = np.cumprod(guess == path[:, :K], axis=1).sum(axis=1)
        step = match + 1
        live = ~done
        committed[live] += step[live]
        rounds[live] += 1
        last = path[np.arange(n), match]
        before = np.where(match > 0, path[np.arange(n), np.maximum(match - 1, 0)], p1)
        p2, p1 = np.where(live, before, p2), np.where(live, last, p1)
        done |= committed >= new_tokens
    return committed, rounds


@dataclass
class ToyResult:
    method: str
    tau: float              # mean over the worlds
    blocks: int             # training blocks per world
    rollouts: int           # target rollouts per world
    secondary_slots: float | None = None


def run_toy(methods=METHODS, worlds=5, seed=0, **world_kwargs):
    """Train every objective in `worlds` toy worlds and decode held-out prompts.

    Returns one ToyResult per method with tau averaged over the worlds.
    """
    taus = {m: [] for m in methods}
    counts, sec = {}, []
    for w in range(worlds):
        world = make_world(seed=seed + w, **world_kwargs)
        V = world.target.shape[0]
        for m in methods:
            anchor, labels, weights, rollouts = training_blocks(world, m, seed=seed + w)
            q = fit_drafter(anchor, labels, weights, V)
            committed, rounds = chain_decode(world, q)
            taus[m].append(committed.sum() / rounds.sum())
            counts.setdefault(m, (len(anchor), rollouts))
            if m == "alr_ira":
                sec.append((weights[rollouts:] > 0).sum(axis=1).mean())
    return [ToyResult(m, float(np.mean(taus[m])), counts[m][0], counts[m][1],
                      float(np.mean(sec)) if m == "alr_ira" else None) for m in methods]


def format_results(results) -> str:
    lines = [f"{'method':10s} {'tau':>6s} {'blocks':>7s} {'rollouts':>9s}"]
    for r in results:
        lines.append(f"{LABELS[r.method]:10s} {r.tau:6.2f} {r.blocks:7d} {r.rollouts:9d}")
    sec = [r.secondary_slots for r in results if r.secondary_slots is not None]
    if sec:
        lines.append(f"slots supervised per IRA secondary block, on average: {sec[0]:.2f}")
    return "\n".join(lines)
