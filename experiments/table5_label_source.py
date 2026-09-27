"""Table 5: label source and survival weighting on fixed UltraChat, three epochs.

MAT is the geometric mean over the seven text benchmarks within a training seed, averaged
over seeds 1 and 2 of each arm, with the sample SD across the two.
"""
from common import TASKS, Check, geomean, mean_sd, text_run, text_tau

# (label source, soft gate, method, T=0, T=1) as printed
PAPER = [("Corpus (KD)", "Off", "kd", "3.586±0.0004", "3.306±0.005"),
         ("Corpus (erase)", "On", "erase", "3.758±0.015", "3.483±0.027"),
         ("Target rollout", "On", "alr_gate", "3.886±0.001", "3.583±0.007"),
         ("Target rollout (ALR)", "Off", "alr", "3.884±0.008", "3.580±0.035")]


def main():
    check = Check()
    print(f"{'label source':22s} {'soft gate':9s} {'T=0':>14s} {'T=1':>14s}")
    for label, gate, m, *printed in PAPER:
        out = []
        for t, p in zip((0, 1), printed):
            mean, sd = p.split("±")
            value = mean_sd(geomean(text_tau(text_run(m, s), k, t) for k in TASKS) for s in (1, 2))
            out.append(check(f"{m} T{t}", value[0], mean) + "±" + check(f"{m} T{t} sd", value[1], sd))
        print(f"{label:22s} {gate:9s} {out[0]:>14s} {out[1]:>14s}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
