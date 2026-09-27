"""Table 2 and Figure 3: one-epoch training against the best erase schedule C, T=0.

tau is rebuilt from data/. Training time is a measured wall-clock and is not rebuilt here.
On ALLaVA it follows from the per-epoch GPU-hours on four H100s that Appendix A reports.
The ShareGPT4V ratios are entered as Table 2 prints them.
"""
from common import Check, find, pm, vision_cell

# GPU-hours for one epoch on the ALLaVA rows (Appendix A), keyed by (method, rollout depth)
ALLAVA_HOURS = {("erase", None): 0.75, ("alr", 14): 1.49, ("alr", 4): 0.95, ("alr_ira", 14): 1.32}
C_ALLAVA = 3 * ALLAVA_HOURS[("erase", None)]           # three epochs of erase
SV_TIME = {("erase", None, 1): 1.00, ("alr", 4, 1): 1.28, ("alr_ira", 14, 1): 1.77}

# (corpus, label, method, depth, epochs, time/C, tau) as printed
PAPER = [
    ("allava", "Erase", "erase", None, 1, "0.33", "3.533±0.006"),
    ("allava", "ALR, R=4", "alr", 4, 1, "0.42", "3.649±0.014"),
    ("allava", "ALR + IRA", "alr_ira", 14, 1, "0.59", "3.728±0.005"),
    ("allava", "Erase (C)", "erase", None, 3, "1.00", "3.539±0.010"),
    ("sharegpt4v", "Erase (C)", "erase", None, 1, "1.00", "3.640±0.002"),
    ("sharegpt4v", "ALR, R=4", "alr", 4, 1, "1.28", "3.695±0.011"),
    ("sharegpt4v", "ALR + IRA", "alr_ira", 14, 1, "1.77", "3.778±0.003"),
]


def time_vs_c(corpus, method, depth, epochs):
    if corpus == "allava":
        return epochs * ALLAVA_HOURS[(method, depth)] / C_ALLAVA
    return SV_TIME[(method, depth, epochs)]


def main():
    check = Check()
    print("Table 2, one epoch against the best erase schedule C, T=0")
    print(f"{'corpus':11s} {'method':10s} {'epochs':>6s} {'time/C':>7s} {'tau':>12s}")
    for corpus, label, method, depth, epochs, t_printed, tau_printed in PAPER:
        cell = vision_cell(find("acceptance", "8b", corpus, method, epochs=epochs, depth=depth))
        mean, sd = tau_printed.split("±")
        check(f"{corpus} {label} tau", cell["avg"][0], mean)
        check(f"{corpus} {label} sd", cell["avg"][1], sd)
        if corpus == "allava":
            tc = check(f"{corpus} {label} time", time_vs_c(corpus, method, depth, epochs), t_printed)
        else:
            tc = f"{time_vs_c(corpus, method, depth, epochs):.2f}"
        print(f"{corpus:11s} {label:10s} {epochs:6d} {tc:>7s} {pm(cell['avg']):>12s}")
    saving = 1 - time_vs_c("allava", "alr_ira", 14, 1)
    print(f"One epoch of ALR + IRA needs {check('saving', 100 * saving, '41')}% less training time than C on ALLaVA")

    print("\nFigure 3, ALLaVA training cost at T=0 (time relative to C from the rounded GPU-hours, MAT)")
    points = {}
    for label, method, depth, epochs in [("Erase", "erase", None, 1), ("Erase", "erase", None, 3),
                                         ("Erase", "erase", None, 6), ("ALR, R=4", "alr", 4, 1),
                                         ("ALR, R=4", "alr", 4, 3), ("ALR, R=14", "alr", 14, 1),
                                         ("ALR, R=14", "alr", 14, 3), ("ALR + IRA", "alr_ira", 14, 1),
                                         ("ALR + IRA", "alr_ira", 14, 3)]:
        tau = vision_cell(find("acceptance", "8b", "allava", method, epochs=epochs, depth=depth))["avg"][0]
        tc = time_vs_c("allava", method, depth, epochs)
        points[(label, epochs)] = (tc, tau)
        print(f"  {label:10s} {epochs} epoch{'s' if epochs > 1 else ' '}  time/C {tc:4.2f}  MAT {tau:.3f}")
    for e in (1, 3):
        ira, alr = points[("ALR + IRA", e)], points[("ALR, R=14", e)]
        assert ira[1] > alr[1] and ira[0] < alr[0]
    assert points[("Erase", 6)][1] < points[("ALR + IRA", 1)][1]
    print("At one and three epochs, ALR + IRA has the higher MAT than full-depth ALR in less time,"
          " and six erase epochs stay below one epoch of ALR + IRA.")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
