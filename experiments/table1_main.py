"""Table 1: speedup over AR and mean acceptance length after three epochs.

Vision tau averages the per-seed geometric mean over COCO, TextVQA and DocVQA (RTX A6000,
three seeds for the 8B target on ALLaVA and two elsewhere). Vision speedups come from the
H100 timing runs and average the runs of one checkpoint. Text values average the two
training seeds of each method on one H100.
"""
import statistics

from common import (NAMES, TASKS, Check, find, geomean, run_speedups, text_run, text_speedup, text_tau,
                    vision_cell)

METHODS = ("dflash", "erase", "alr", "alr_ira")
SETTINGS = (("8b", "allava", "Qwen3-VL-8B, ALLaVA"), ("8b", "sharegpt4v", "Qwen3-VL-8B, ShareGPT4V"),
            ("4b", "allava", "Qwen3-VL-4B, ALLaVA"))

# (speedup, tau) as printed, by temperature, setting and method
PAPER_VISION = {
    0: {("8b", "allava"): [("2.09", "2.88"), ("2.62", "3.54"), ("2.68", "3.70"), ("2.82", "3.84")],
        ("8b", "sharegpt4v"): [("2.01", "2.80"), ("2.62", "3.57"), ("2.80", "3.67"), ("2.77", "3.83")],
        ("4b", "allava"): [("2.19", "2.92"), ("2.62", "3.56"), ("2.65", "3.63"), ("2.77", "3.81")]},
    1: {("8b", "allava"): [("2.04", "2.79"), ("2.49", "3.39"), ("2.55", "3.50"), ("2.68", "3.65")],
        ("8b", "sharegpt4v"): [("1.90", "2.71"), ("2.54", "3.39"), ("2.54", "3.48"), ("2.58", "3.60")],
        ("4b", "allava"): [("2.04", "2.78"), ("2.45", "3.33"), ("2.53", "3.41"), ("2.67", "3.53")]},
}
PAPER_TEXT = {
    0: {"mt-bench": "1.82 2.50 2.07 2.85 2.09 2.94 2.13 2.99", "alpaca": "1.56 2.17 1.88 2.52 1.87 2.58 1.92 2.62",
        "gsm8k": "2.79 3.96 3.16 4.48 3.38 4.76 3.55 4.83", "aime24": "2.61 3.69 2.85 4.06 2.91 4.20 2.97 4.27",
        "aime25": "2.76 4.07 3.01 4.34 2.94 4.53 3.14 4.66", "humaneval": "2.98 4.22 3.46 4.92 3.56 4.98 3.63 5.02",
        "livecodebench": "2.33 3.25 2.64 3.72 2.62 3.81 2.73 3.84", "avg": "2.35 3.32 2.67 3.75 2.70 3.87 2.80 3.93"},
    1: {"mt-bench": "1.79 2.42 1.97 2.72 2.06 2.79 2.15 2.83", "alpaca": "1.52 2.16 1.72 2.47 1.78 2.54 1.80 2.57",
        "gsm8k": "2.71 3.78 3.01 4.26 3.19 4.50 3.29 4.55", "aime24": "2.41 3.35 2.49 3.54 2.58 3.66 2.59 3.65",
        "aime25": "2.40 3.49 2.62 3.77 2.46 3.85 2.66 3.96", "humaneval": "2.89 3.97 3.22 4.57 3.19 4.63 3.20 4.66",
        "livecodebench": "2.20 3.00 2.44 3.38 2.42 3.47 2.52 3.49", "avg": "2.23 3.10 2.45 3.46 2.48 3.56 2.55 3.60"},
}
TEXT_NAMES = {"mt-bench": "MT-Bench", "alpaca": "Alpaca", "gsm8k": "GSM8K", "aime24": "AIME24", "aime25": "AIME25",
              "humaneval": "HumanEval", "livecodebench": "LiveCodeBench", "avg": "Avg. (7 tasks)"}


def timing_runs(target, corpus, method, t):
    """The H100 runs behind a speed cell. Session 1 timed the ALLaVA 8B drafters and the ShareGPT4V
    ALR + IRA drafter at T=0, session 2 everything else."""
    runs = find("timing", target, corpus, method, temperature=t, session=1) if t == 0 else []
    return [r for r in runs if "regen" not in r] or find("timing", target, corpus, method, temperature=t, session=2)


def vision_speed(target, corpus, method, t):
    return statistics.mean(run_speedups(r)[0] for r in timing_runs(target, corpus, method, t))


def main():
    check = Check()
    for t in (0, 1):
        print(f"\nVision, T={t}    speedup and tau per setting")
        print(f"{'method':12s}" + "".join(f" | {label:>24s}" for _, _, label in SETTINGS))
        for i, m in enumerate(METHODS):
            line = f"{NAMES[m]:12s}"
            for target, corpus, _ in SETTINGS:
                sp, tau = PAPER_VISION[t][(target, corpus)][i]
                cell = vision_cell(find("acceptance", target, corpus, m, temperature=t))
                s = check(f"T{t} {target} {corpus} {m} speedup", vision_speed(target, corpus, m, t), sp)
                a = check(f"T{t} {target} {corpus} {m} tau", cell["avg"][0], tau)
                line += f" | {s + 'x':>12s} {a:>11s}"
            print(line)
    for t in (0, 1):
        print(f"\nText, Qwen3-4B, T={t}    speedup and tau, mean of seeds 0 and 1")
        print(f"{'benchmark':15s}" + "".join(f" | {NAMES[m]:>15s}" for m in METHODS))
        for task in TASKS + ("avg",):
            printed = PAPER_TEXT[t][task].split()
            line = f"{TEXT_NAMES[task]:15s}"
            for i, m in enumerate(METHODS):
                runs = [text_run(m, s) for s in (0, 1)]
                if task == "avg":
                    sp = statistics.mean(geomean(text_speedup(r, k, t) for k in TASKS) for r in runs)
                    tau = statistics.mean(geomean(text_tau(r, k, t) for k in TASKS) for r in runs)
                else:
                    sp = statistics.mean(text_speedup(r, task, t) for r in runs)
                    tau = statistics.mean(text_tau(r, task, t) for r in runs)
                s = check(f"text T{t} {task} {m} speedup", sp, printed[2 * i])
                a = check(f"text T{t} {task} {m} tau", tau, printed[2 * i + 1])
                line += f" | {s + 'x':>7s} {a:>7s}"
            print(line)
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
