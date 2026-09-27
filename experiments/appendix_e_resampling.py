"""Appendix E: the comparison thresholds and the prompt-resampling intervals.

The thresholds follow from the 1.66% tolerance and the pooled seed SD of 0.35%. The
intervals resample the 99 scored prompts of each domain with replacement 10,000 times
(generator seed 0), apply the same resample to both arms and every seed, and take the
2.5 and 97.5 percentiles of ALR + IRA's relative gain over the erase reference.
"""
import math

import numpy as np

from common import DOMAINS, Check, find, vision_prompts

TOLERANCE, SEED_SD = 1.66, 0.35
DRAWS = 10000
# (label, target, corpus, erase epochs, interval as printed)
PAIRS = [("ALLaVA", "8b", "allava", 3, ("7.71", "9.36")),
         ("ShareGPT4V", "8b", "sharegpt4v", 1, ("4.39", "5.79")),
         ("smaller target", "4b", "allava", 3, ("6.36", "7.91"))]


def arrays(runs):
    p = vision_prompts()
    return [{d: (np.array(p[r][d]["committed"], float), np.array(p[r][d]["rounds"], float)) for d in DOMAINS}
            for r in runs]


def value(arm, idx=None):
    per_run = []
    for r in arm:
        logs = 0.0
        for d in DOMAINS:
            c, w = r[d]
            if idx is None:
                logs = logs + np.log(c.sum() / w.sum())
            else:
                logs = logs + np.log(c[idx[d]].sum(axis=1) / w[idx[d]].sum(axis=1))
        per_run.append(np.exp(logs / len(DOMAINS)))
    return np.mean(per_run, axis=0)


def main():
    check = Check()
    print("Comparison thresholds (%), improvement supported above the upper one")
    for n, lo_p, hi_p in ((2, "0.96", "2.36"), (3, "1.09", "2.23")):
        sd_diff = SEED_SD * math.sqrt(2 / n)
        lo = check(f"n{n} lower", TOLERANCE - 2 * sd_diff, lo_p)
        hi = check(f"n{n} upper", TOLERANCE + 2 * sd_diff, hi_p)
        print(f"  {n} training replicates per method  lower {lo}  upper {hi}")

    print("\nALR + IRA over erase, 95% prompt-resampling interval (%)")
    for label, target, corpus, epochs, (lo_p, hi_p) in PAIRS:
        a = arrays(find("acceptance", target, corpus, "alr_ira"))
        b = arrays(find("acceptance", target, corpus, "erase", epochs=epochs))
        n = {d: len(a[0][d][0]) for d in DOMAINS}
        rng = np.random.default_rng(0)
        idx = {d: rng.integers(0, n[d], size=(DRAWS, n[d])) for d in DOMAINS}
        point = (value(a) / value(b) - 1) * 100
        lo, hi = np.percentile((value(a, idx) / value(b, idx) - 1) * 100, [2.5, 97.5])
        print(f"  {label:15s} point {point:+.2f}  interval {check(label + ' lo', lo, lo_p)}"
              f" to {check(label + ' hi', hi, hi_p)}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
