"""Tables 9 and 10: text speedup and tau per benchmark, mean and SD across the two training seeds.

Qwen3-4B trained on the fixed UltraChat subset, one H100. Every method shares the AR timing
of each benchmark and temperature.
"""
from common import NAMES, Check, mean_sd, text_run, text_speedup, text_tau

METHODS = ("dflash", "erase", "alr", "alr_ira")
# speedup and tau for DFlash, erase, ALR, ALR + IRA
PAPER = {
    ("mt-bench", 0): "1.82±0.07 2.50±0.01 2.07±0.04 2.85±0.01 2.09±0.01 2.94±0.02 2.13±0.02 2.99±0.00",
    ("gsm8k", 0): "2.79±0.05 3.96±0.01 3.16±0.08 4.48±0.01 3.38±0.09 4.76±0.00 3.55±0.06 4.83±0.07",
    ("humaneval", 0): "2.98±0.07 4.22±0.02 3.46±0.10 4.92±0.00 3.56±0.13 4.98±0.06 3.63±0.00 5.02±0.06",
    ("mt-bench", 1): "1.79±0.03 2.42±0.05 1.97±0.04 2.72±0.04 2.06±0.03 2.79±0.02 2.15±0.04 2.83±0.01",
    ("gsm8k", 1): "2.71±0.05 3.78±0.01 3.01±0.09 4.26±0.01 3.19±0.08 4.50±0.03 3.29±0.05 4.55±0.02",
    ("humaneval", 1): "2.89±0.02 3.97±0.07 3.22±0.02 4.57±0.10 3.19±0.07 4.63±0.00 3.20±0.11 4.66±0.01",
    ("alpaca", 0): "1.56±0.02 2.17±0.00 1.88±0.01 2.52±0.01 1.87±0.07 2.58±0.02 1.92±0.04 2.62±0.00",
    ("alpaca", 1): "1.52±0.04 2.16±0.01 1.72±0.09 2.47±0.01 1.78±0.13 2.54±0.01 1.80±0.08 2.57±0.00",
    ("aime24", 0): "2.61±0.02 3.69±0.03 2.85±0.04 4.06±0.01 2.91±0.07 4.20±0.04 2.97±0.11 4.27±0.05",
    ("aime24", 1): "2.41±0.03 3.35±0.05 2.49±0.04 3.54±0.07 2.58±0.14 3.66±0.05 2.59±0.02 3.65±0.04",
    ("aime25", 0): "2.76±0.14 4.07±0.04 3.01±0.03 4.34±0.02 2.94±0.04 4.53±0.08 3.14±0.01 4.66±0.06",
    ("aime25", 1): "2.40±0.04 3.49±0.13 2.62±0.05 3.77±0.06 2.46±0.03 3.85±0.06 2.66±0.04 3.96±0.01",
    ("livecodebench", 0): "2.33±0.01 3.25±0.05 2.64±0.06 3.72±0.01 2.62±0.11 3.81±0.03 2.73±0.13 3.84±0.02",
    ("livecodebench", 1): "2.20±0.06 3.00±0.04 2.44±0.00 3.38±0.02 2.42±0.09 3.47±0.01 2.52±0.15 3.49±0.01",
}
TABLES = (("Table 9", ("mt-bench", "gsm8k", "humaneval")),
          ("Table 10", ("alpaca", "aime24", "aime25", "livecodebench")))


def main():
    check = Check()
    for title, tasks in TABLES:
        for task in tasks:
            for t in (0, 1):
                printed = PAPER[(task, t)].split()
                print(f"\n{title}, {task}, T={t}")
                for i, m in enumerate(METHODS):
                    runs = [text_run(m, s) for s in (0, 1)]
                    sp = mean_sd(text_speedup(r, task, t) for r in runs)
                    tau = mean_sd(text_tau(r, task, t) for r in runs)
                    out = []
                    for label, value, p in (("speedup", sp, printed[2 * i]), ("tau", tau, printed[2 * i + 1])):
                        mean, sd = p.split("±")
                        out.append(check(f"{task} T{t} {m} {label}", value[0], mean) + "±"
                                   + check(f"{task} T{t} {m} {label} sd", value[1], sd))
                    print(f"  {NAMES[m]:10s} speedup {out[0]}x   tau {out[1]}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
