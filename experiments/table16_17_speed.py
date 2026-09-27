"""Tables 16 and 17: decoding speedup over AR on one H100, T=0 unless noted.

A domain's speedup is mean AR milliseconds per token over the drafter's in the same run.
A row's speedup is the geometric mean over the three domains, averaged over its timing
runs. MAT (n) in Table 16 is the A6000 value of Table 8 with its seed count, while Table 17
reports the MAT of the H100 runs themselves.
"""
import statistics

from common import DOMAINS, NAMES, Check, find, geomean, run_speedups, vision_cell, vision_prompts, welch

# (label, corpus, method, epochs, depth, speedup, per-domain, runs, MAT, n) as printed
TABLE16 = [
    ("DFlash, iso-data", "allava", "dflash", 3, "default", "2.09", "1.86 1.99 2.46", "1", "2.878", 3),
    ("Erase (PARD-2 soft survival weight)", "allava", "erase", 3, "default", "2.62", "2.28 2.55 3.10", "3 (0.03)",
     "3.539", 3),
    ("ALR", "allava", "alr", 3, "default", "2.68", "2.22 2.77 3.13", "1", "3.696", 3),
    ("ALR, rollout depth R=4, one epoch", "allava", "alr", 1, 4, "2.77", "2.34 2.71 3.37", "1", "3.649", 3),
    ("ALR + IRA, one epoch", "allava", "alr_ira", 1, "default", "2.69", "2.27 2.75 3.11", "1", "3.728", 3),
    ("ALR + IRA", "allava", "alr_ira", 3, "default", "2.82", "2.46 2.95 3.08", "4 (0.03)", "3.841", 3),
    ("Regeneration, truncated", "allava", "regen_truncated", 3, "default", "2.89", "2.49 3.01 3.23", "1",
     "3.833", 2),
    ("Erase, one epoch, ShareGPT4V", "sharegpt4v", "erase", 1, "default", "2.66", "2.46 2.55 3.01", "1",
     "3.640", 2),
    ("ALR + IRA, ShareGPT4V", "sharegpt4v", "alr_ira", 3, "default", "2.77", "2.72 2.67 2.93", "1", "3.825", 2),
]
# Table 17, (speedup, tau) at T=0 and T=1
TABLE17 = {"eagle3": ("1.27", "2.32", "1.24", "2.29"), "dflash": ("2.09", "2.87", "2.04", "2.78"),
           "erase": ("2.65", "3.54", "2.49", "3.38"), "alr": ("2.68", "3.69", "2.55", "3.47"),
           "alr_ira": ("2.85", "3.84", "2.68", "3.63")}


def h100_tau(run):
    p = vision_prompts()[run]
    return geomean(sum(p[d]["committed"]) / sum(p[d]["rounds"]) for d in DOMAINS)


def main():
    check = Check()
    print("Table 16, one H100, session 1")
    print(f"{'row':38s} {'speedup':>7s} {'caption':>8s} {'textvqa':>8s} {'docvqa':>8s} {'runs':>9s} {'MAT (n)':>10s}")
    speeds = {}
    for label, corpus, method, epochs, depth, sp, per, nruns, mat_p, n in TABLE16:
        runs = find("timing", "8b", corpus, method, epochs=epochs, depth=depth, session=1)
        sp_runs = [run_speedups(r)[0] for r in runs]
        speeds[label] = sp_runs
        s = check(f"{label} speedup", statistics.mean(sp_runs), sp)
        doms = [check(f"{label} {d}", statistics.mean(run_speedups(r)[1][d] for r in runs), p)
                for d, p in zip(DOMAINS, per.split())]
        runs_str = str(len(runs))
        if len(runs) > 1:
            runs_str += f" ({check(f'{label} run sd', statistics.stdev(sp_runs), nruns.split('(')[1][:-1])})"
        assert runs_str == nruns
        cell = vision_cell(find("acceptance", "8b", corpus, method, epochs=epochs, depth=depth))
        assert cell["n"] == n
        m = check(f"{label} MAT", cell["avg"][0], mat_p)
        mat_n = f"{m} ({n})"
        print(f"{label:38s} {s + 'x':>7s} " + " ".join(f"{d:>8s}" for d in doms) + f" {runs_str:>9s} {mat_n:>10s}")

    d, lo, hi = welch(speeds["ALR + IRA"], speeds["Erase (PARD-2 soft survival weight)"])
    print(f"ALR + IRA over erase, ALLaVA  {check('welch d', d, '0.20')}x, 95% Welch interval "
          f"{check('welch lo', lo, '0.13')} to {check('welch hi', hi, '0.26')}x")
    first, last = (run_speedups(f"session1-8b-allava-alr_ira-e3-s0-t0-run{k}")[0] for k in (1, 2))
    print(f"ALR + IRA timed at the start and end of the session drifts by "
          f"{check('drift', abs(last / first - 1) * 100, '0.62')}%")
    sv = statistics.mean(speeds["ALR + IRA, ShareGPT4V"]) / statistics.mean(speeds["Erase, one epoch, ShareGPT4V"])
    print(f"ShareGPT4V, ALR + IRA over one-epoch erase  +{check('sv', (sv - 1) * 100, '4.1')}%")

    print("\nTable 17, one H100, Qwen3-VL-8B trained on ALLaVA")
    print(f"{'method':14s} {'T=0 speedup':>11s} {'tau':>5s} {'T=1 speedup':>12s} {'tau':>5s}")
    for method, printed in TABLE17.items():
        out = []
        for t in (0, 1):
            if method == "eagle3":
                runs = find("timing", "8b", "allava", "eagle3", epochs=None, depth=None, temperature=t,
                            decoder="eagle3_tree", session=3)
            elif t == 0:
                runs = [r for r in find("timing", "8b", "allava", method, session=1) if r.endswith("run1")]
            else:
                runs = find("timing", "8b", "allava", method, temperature=1, session=2)
            assert len(runs) == 1, runs
            out.append(check(f"{method} T{t} speedup", run_speedups(runs[0])[0], printed[2 * t]))
            out.append(check(f"{method} T{t} tau", h100_tau(runs[0]), printed[2 * t + 1]))
        print(f"{NAMES[method]:14s} {out[0] + 'x':>11s} {out[1]:>5s} {out[2] + 'x':>12s} {out[3]:>5s}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
