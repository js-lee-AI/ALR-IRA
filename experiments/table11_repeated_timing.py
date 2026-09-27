"""Table 11: repeated A100 timing of AR, erase and ALR + IRA on 220 fixed text prompts.

Milliseconds per token are averaged over the three timing repeats and then over the prompts
of each task. Speedups are geometric means of task latency ratios within each checkpoint
pair, averaged over the two pairs. The intervals resample prompts within each task 10,000
times, keeping methods, checkpoints, temperatures and repeats together.
"""
import csv

import numpy as np

from common import DATA, TASKS, Check

METHODS = ("ar", "erase", "alr_ira")
METRICS = ("total", "decode")
DRAWS, SEED = 10000, 20260926
PAPER = {0: ("2.66", "2.79", "4.62", "3.80", "5.42"), 1: ("2.48", "2.58", "3.89", "2.52", "5.28")}


def load():
    rows = {}
    with open(DATA / "text" / "a100_timing.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["task"], []).append(r)
    arrays = {}
    for task in TASKS:
        n = 1 + max(int(r["prompt"]) for r in rows[task])
        v = np.full((n, 2, 3, 2, 3, 2), np.nan)    # prompt, checkpoint pair, repeat, T, method, metric
        for r in rows[task]:
            for m, method in enumerate(METHODS):
                for k, metric in enumerate(METRICS):
                    v[int(r["prompt"]), int(r["seed"]) - 1, int(r["repeat"]), int(r["temperature"]), m, k] = \
                        float(r[f"{method}_{metric}_ms"])
        assert np.isfinite(v).all()
        arrays[task] = v
    return arrays


def main():
    check = Check()
    arrays = load()
    task_means = np.stack([arrays[t].mean(axis=(0, 2)) for t in TASKS])   # task, pair, T, method, metric
    speedups = np.exp(np.log(task_means[..., 0:1, :] / task_means).mean(axis=0)).mean(axis=0)
    contrasts = np.exp(np.log(task_means[..., 1, :] / task_means[..., 2, :]).mean(axis=0)).mean(axis=0)

    rng = np.random.default_rng(SEED)
    boots = np.empty((DRAWS, len(TASKS), 2, 2, 3, 2))
    for j, task in enumerate(TASKS):
        per_prompt = arrays[task].mean(axis=2)
        n = len(per_prompt)
        for start in range(0, DRAWS, 500):
            end = min(start + 500, DRAWS)
            boots[start:end, j] = per_prompt[rng.integers(0, n, (end - start, n))].mean(axis=1)
    ratio = np.exp(np.log(boots[..., 1, :] / boots[..., 2, :]).mean(axis=1)).mean(axis=1)
    lo, hi = np.quantile(ratio, [0.025, 0.975], axis=0)

    prompts = sum(len(a) for a in arrays.values())
    print(f"{prompts} prompts, total generation milliseconds per token (decode-only in brackets)")
    print(f"{'':20s} {'T=0':>22s} {'T=1':>22s}")
    lines = {"Erase": [], "ALR + IRA": [], "Gain over erase (%)": []}
    for t in (0, 1):
        p = PAPER[t]
        lines["Erase"].append(check(f"T{t} erase", speedups[t, 1, 0], p[0]) + f"x ({speedups[t, 1, 1]:.2f}x)")
        lines["ALR + IRA"].append(check(f"T{t} ira", speedups[t, 2, 0], p[1]) + f"x ({speedups[t, 2, 1]:.2f}x)")
        g = check(f"T{t} gain", 100 * (contrasts[t, 0] - 1), p[2])
        a = check(f"T{t} lo", 100 * (lo[t, 0] - 1), p[3])
        b = check(f"T{t} hi", 100 * (hi[t, 0] - 1), p[4])
        lines["Gain over erase (%)"].append(f"{g} [{a}, {b}]")
    for label, cells in lines.items():
        print(f"{label:20s} {cells[0]:>22s} {cells[1]:>22s}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
