"""Figure 2 and the headline numbers of Sections 1 and 4.

Per benchmark MAT gain over erase, from means across training replicates, for the three
vision-language settings and the seven text benchmarks, and the ALLaVA decoding speedup
by domain. Pass --plot to draw the figure (needs matplotlib).
"""
import argparse
import statistics

from common import (DOMAINS, NAMES, TASKS, Check, find, gain, run_speedups, text_run, text_tau, vision_cell)
from table1_main import timing_runs

METHODS = ("dflash", "alr", "alr_ira")
SETTINGS = (("8b", "allava", "ALLaVA, 8B"), ("8b", "sharegpt4v", "ShareGPT4V, 8B"), ("4b", "allava", "ALLaVA, 4B"))


def vision_means(target, corpus, method, t):
    c = vision_cell(find("acceptance", target, corpus, method, temperature=t))
    return {**{d: c[d][0] for d in DOMAINS}, "avg": c["avg"][0]}


def text_means(method, t):
    runs = [text_run(method, s) for s in (0, 1)]
    return {k: statistics.mean(text_tau(r, k, t) for r in runs) for k in TASKS}


def gains(t):
    out = {}
    for target, corpus, label in SETTINGS:
        ref = vision_means(target, corpus, "erase", t)
        out[label] = {m: {d: gain(vision_means(target, corpus, m, t)[d], ref[d]) for d in DOMAINS} for m in METHODS}
    ref = text_means("erase", t)
    out["Text, Qwen3-4B"] = {m: {k: gain(text_means(m, t)[k], ref[k]) for k in TASKS} for m in METHODS}
    return out


def speed_by_domain(method, t):
    runs = timing_runs("8b", "allava", method, t)
    per = {d: statistics.mean(run_speedups(r)[1][d] for r in runs) for d in DOMAINS}
    per["geomean"] = statistics.mean(run_speedups(r)[0] for r in runs)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", help="write the figure to this file")
    args = ap.parse_args()
    check = Check()
    all_gains = {}
    for t in (0, 1):
        all_gains[t] = g = gains(t)
        print(f"\nT={t}, MAT gain over erase (%)")
        for panel, rows in g.items():
            keys = list(next(iter(rows.values())))
            print(f"  {panel:16s}" + "".join(f" {k[:9]:>9s}" for k in keys))
            for m in METHODS:
                print(f"  {NAMES[m]:16s}" + "".join(f" {rows[m][k]:+9.2f}" for k in keys))
        print(f"  ALLaVA speedup over AR by domain, T={t}")
        for m in ("dflash", "erase", "alr", "alr_ira"):
            s = speed_by_domain(m, t)
            print(f"  {NAMES[m]:16s}" + "".join(f" {s[k]:8.2f}x" for k in s))

    print("\nHeadline numbers")
    for t, printed in ((0, "36.5"), (1, "33.1")):
        best = max(gain(vision_means("8b", c, "alr_ira", t)["avg"], vision_means("8b", c, "dflash", t)["avg"])
                   for c in ("allava", "sharegpt4v"))
        print(f"  ALR + IRA over DFlash, best 8B setting, T={t}  {check(f'dflash T{t}', best, printed)}%")
    for label, target, corpus, printed in (("ALLaVA, 8B", "8b", "allava", "8.54"),
                                           ("ALLaVA, 4B", "4b", "allava", "7.13")):
        g = gain(vision_means(target, corpus, "alr_ira", 0)["avg"], vision_means(target, corpus, "erase", 0)["avg"])
        print(f"  ALR + IRA over erase, {label}, T=0  {check(label, g, printed)}%")
    ira = vision_cell(find("acceptance", "8b", "sharegpt4v", "alr_ira"))["avg"][0]
    c_sv = vision_cell(find("acceptance", "8b", "sharegpt4v", "erase", epochs=1))["avg"][0]
    print(f"  ALR + IRA over one-epoch erase, ShareGPT4V, T=0  {check('sv lead', gain(ira, c_sv), '5.09')}%")
    for t, printed in ((0, "4.96"), (1, "4.10")):
        agg = {m: statistics.mean(
            statistics.geometric_mean(text_tau(text_run(m, s), k, t) for k in TASKS) for s in (0, 1))
            for m in ("erase", "alr_ira")}
        print(f"  ALR + IRA over erase, text aggregate, T={t}  "
              f"{check(f'text T{t}', gain(agg['alr_ira'], agg['erase']), printed)}%")
    every = all(v > 0 for t in (0, 1) for rows in all_gains[t].values() for v in rows["alr_ira"].values())
    doc = all(all_gains[t]["ALLaVA, 4B"]["alr"]["docvqa"] < 0 < all_gains[t]["ALLaVA, 4B"]["alr_ira"]["docvqa"]
              for t in (0, 1))
    assert every and doc
    print("  ALR + IRA is above erase on every domain and benchmark at both temperatures, and on the 4B"
          " target's DocVQA ALR alone is below erase while ALR + IRA is above it.")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
        for ax, t in zip(axes, (0, 1)):
            x, labels = 0, []
            for panel, rows in all_gains[t].items():
                for k in rows["alr"]:
                    for i, m in enumerate(METHODS):
                        if m != "dflash":
                            ax.bar(x + (i - 1.5) * 0.35, rows[m][k], 0.35, color=("C1", "C2")[i - 1])
                    labels.append((x, k))
                    x += 1
                x += 0.5
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks([p for p, _ in labels], [k for _, k in labels], rotation=45, ha="right")
            ax.set_ylabel(f"MAT gain over erase (%), T={t}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
