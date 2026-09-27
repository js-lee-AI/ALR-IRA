"""Loaders for data/ and the check that compares each rebuilt value with the printed one."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from alr_ira.metrics import geomean, mat  # noqa: E402

DOMAINS = ("caption", "textvqa", "docvqa")
TASKS = ("mt-bench", "alpaca", "gsm8k", "aime24", "aime25", "humaneval", "livecodebench")
NAMES = {"dflash": "DFlash", "kd": "KD", "erase": "Erase", "erase_hard": "Erase, hard gate", "alr": "ALR",
         "alr_gate": "ALR with the soft gate", "alr_ira": "ALR + IRA", "slot_control": "Slot control",
         "regen_truncated": "Regeneration, truncated", "regen_window": "Regeneration, loss window",
         "eagle3": "EAGLE-3 (HF)"}


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# vision

@lru_cache(maxsize=None)
def vision_runs():
    runs = {}
    for r in _rows(DATA / "vision" / "runs.csv"):
        r["epochs"] = float(r["epochs"]) if r["epochs"] else None
        r["rollout_depth"] = int(r["rollout_depth"]) if r["rollout_depth"] else None
        r["gamma"] = float(r["gamma"]) if r["gamma"] else None
        for k in ("data_order", "seed", "temperature"):
            r[k] = int(r[k])
        r["session"] = int(r["session"]) if r["session"] else None
        runs[r["run"]] = r
    return runs


@lru_cache(maxsize=None)
def vision_prompts():
    """run -> domain -> dict of per-prompt lists (committed, rounds and, for timing runs, ar_ms, draft_ms)."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name in ("acceptance", "timing"):
        for r in _rows(DATA / "vision" / f"{name}.csv"):
            d = out[r["run"]][r["domain"]]
            d["committed"].append(int(r["committed"]))
            d["rounds"].append(int(r["rounds"]))
            if name == "timing":
                d["ar_ms"].append(float(r["ar_ms"]))
                d["draft_ms"].append(float(r["draft_ms"]))
    return out


def find(set="acceptance", target="8b", corpus="allava", method=None, epochs=3, depth="default", gamma=None,
         order=0, temperature=0, decoder="tree", session=None, seeds=None):
    """Run ids matching a setting, ordered by seed then name. depth='default' means 14 for rollout methods."""
    out = []
    for r in vision_runs().values():
        if r["set"] != set or r["target"] != target or r["corpus"] != corpus or r["method"] != method:
            continue
        if epochs is not None and r["epochs"] != epochs:
            continue
        want_depth = (14 if method in ("alr", "alr_ira", "slot_control") else None) if depth == "default" else depth
        if r["rollout_depth"] != want_depth:
            continue
        default_gamma = {"dflash": 7.0, "eagle3": None}.get(method, 2.0)
        want_gamma = gamma if gamma is not None else default_gamma
        if r["gamma"] != want_gamma:
            continue
        if r["data_order"] != order or r["temperature"] != temperature or r["decoder"] != decoder:
            continue
        if session is not None and r["session"] != session:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        out.append(r["run"])
    return sorted(out, key=lambda k: (vision_runs()[k]["seed"], k))


def domain_taus(run):
    p = vision_prompts()[run]
    return {d: mat(p[d]["committed"], p[d]["rounds"]) for d in DOMAINS}


def run_speedups(run):
    """Per-domain speedup of one timing run, and their geometric mean."""
    p = vision_prompts()[run]
    per = {d: statistics.mean(p[d]["ar_ms"]) / statistics.mean(p[d]["draft_ms"]) for d in DOMAINS}
    return geomean(per.values()), per


def mean_sd(values):
    values = list(values)
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def vision_cell(runs):
    """Mean and sample SD across training seeds, per domain and for the geometric mean over domains."""
    if not runs:
        raise ValueError("no runs for this setting")
    taus = [domain_taus(r) for r in runs]
    cell = {d: mean_sd(t[d] for t in taus) for d in DOMAINS}
    cell["avg"] = mean_sd(geomean(t.values()) for t in taus)
    cell["n"] = len(runs)
    return cell


# text

@lru_cache(maxsize=None)
def text_runs():
    return {r["run"]: dict(r, seed=int(r["seed"])) for r in _rows(DATA / "text" / "runs.csv")}


@lru_cache(maxsize=None)
def text_prompts():
    """run -> (task, temperature) -> dict of per-prompt lists."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _rows(DATA / "text" / "prompts.csv"):
        d = out[r["run"]][(r["task"], int(r["temperature"]))]
        d["committed"].append(int(r["committed"]))
        d["rounds"].append(int(r["rounds"]))
        if r["draft_ms"]:
            d["draft_ms"].append(float(r["draft_ms"]))
    return out


@lru_cache(maxsize=None)
def text_ar():
    out = defaultdict(list)
    for r in _rows(DATA / "text" / "ar_reference.csv"):
        out[(r["task"], int(r["temperature"]))].append(float(r["ar_ms"]))
    return out


def text_tau(run, task, t):
    p = text_prompts()[run][(task, t)]
    return mat(p["committed"], p["rounds"])


def text_speedup(run, task, t):
    """Mean AR milliseconds per token over the drafter's, with the shared AR reference of the task."""
    return statistics.mean(text_ar()[(task, t)]) / statistics.mean(text_prompts()[run][(task, t)]["draft_ms"])


def text_run(method, seed):
    run = f"{method}-s{seed}"
    if run not in text_runs():
        raise KeyError(run)
    return run


# comparison with the paper

class Check:
    """Formats each rebuilt value as the paper prints it and compares the two strings."""

    def __init__(self):
        self.n = 0
        self.bad = []

    def __call__(self, label, value, printed, digits=None):
        if digits is None:
            digits = len(printed.split(".")[1]) if "." in printed else 0
        got = f"{value:+.{digits}f}" if printed.startswith(("+", "-")) else f"{value:.{digits}f}"
        self.n += 1
        if got != printed:
            self.bad.append(f"{label}: rebuilt {got}, paper {printed}")
        return got

    def done(self):
        for line in self.bad:
            print("MISMATCH", line)
        print(f"{self.n - len(self.bad)} of {self.n} printed values reproduced")
        return 1 if self.bad else 0


def pm(cell, digits=3):
    m, s = cell
    return f"{m:.{digits}f}" + ("" if s is None else f"±{s:.{digits}f}")


def gain(new, ref):
    return (new / ref - 1) * 100


def log_mean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def _betacf(a, b, x):
    # continued fraction of the regularized incomplete beta function (modified Lentz)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_quantile(p, df):
    """Quantile of Student's t for p > 0.5, by bisection on the CDF."""
    def cdf(t):
        return 1.0 - 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))

    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def welch(a, b):
    """Difference of means and its 95% Welch interval."""
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t_quantile(0.975, df) * math.sqrt(va + vb)
    return ma - mb, ma - mb - half, ma - mb + half
