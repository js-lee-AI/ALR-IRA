"""Table 4: length-matched target regeneration on the ALLaVA prompts, three epochs, T=0.

(a) overall MAT with Qwen3-VL-8B, mean over training seeds. (b) MAT gain over ALR on COCO
captions by the token offset at which a verification round starts, from one checkpoint of
each method (data/vision/rounds.csv).
"""
import csv

from common import DATA, NAMES, Check, find, gain, vision_cell

BINS = ((0, 16, "0-15"), (16, 64, "16-63"), (64, 128, "64-127"), (128, None, "128+"))
PAPER = {"alr": ("3.696", "0.00 0.00 0.00 0.00"), "alr_ira": ("3.841", "+1.32 +5.60 +5.15 +3.57"),
         "regen_truncated": ("3.833", "+1.49 +5.61 +4.51 +4.72"),
         "regen_window": ("3.856", "+3.60 +7.21 +5.12 +5.65")}


def binned_mat(run):
    s, n = [0] * len(BINS), [0] * len(BINS)
    with open(DATA / "vision" / "rounds.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["run"] != run:
                continue
            start = int(r["start"])
            k = next(i for i, (lo, hi, _) in enumerate(BINS) if start >= lo and (hi is None or start < hi))
            s[k] += int(r["committed"])
            n[k] += 1
    return [a / b for a, b in zip(s, n)]


def main():
    check = Check()
    overall, ira_gains = {}, None
    ref = binned_mat("8b-allava-alr-e3-s0-t0")
    print("(a) overall MAT, and (b) COCO gain over ALR in % by the offset where a round starts")
    print(f"{'method':26s} {'seeds':>5s} {'MAT':>6s} |" + " ".join(f"{b[2]:>7s}" for b in BINS))
    for m, (mat_printed, bins_printed) in PAPER.items():
        cell = vision_cell(find("acceptance", "8b", "allava", m))
        overall[m] = cell["avg"][0]
        a = check(f"{m} MAT", overall[m], mat_printed)
        g = binned_mat(f"8b-allava-{m}-e3-s0-t0")
        gains = [check(f"{m} bin {b[2]}", gain(x, y), p, 2) if m != "alr" else f"{0:.2f}"
                 for x, y, b, p in zip(g, ref, BINS, bins_printed.split())]
        print(f"{NAMES[m]:26s} {cell['n']:5d} {a:>6s} |" + " ".join(f"{x:>7s}" for x in gains))
        if m == "alr_ira":
            ira_gains = [gain(x, y) for x, y in zip(g, ref)]
    rise = sum(ira_gains[1:]) / 3 - ira_gains[0]
    print(f"IRA's mean gain in the last three bins exceeds the first by {check('rise', rise, '3.45')} points")
    for m in ("regen_truncated", "regen_window"):
        d = gain(overall["alr_ira"], overall[m])
        assert abs(d) < 0.4
        print(f"ALR + IRA against {NAMES[m].lower()}: {d:+.2f}% overall MAT, within 0.4%")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
