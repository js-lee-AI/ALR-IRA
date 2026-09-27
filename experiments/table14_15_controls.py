"""Tables 14 and 15, Figure 4a and the erase-comparator check of Appendix G. ALLaVA, three epochs.

Every value is a relative difference of the MAT aggregate (the per-seed geometric mean over
COCO, TextVQA and DocVQA, averaged over seeds). The common seeds of Table 14 are 0 and 1.
"""
from common import Check, find, gain, pm, vision_cell


def avg(runs):
    return vision_cell(runs)["avg"][0]


def runs(method, seeds=None, **kw):
    return find("acceptance", "8b", "allava", method, seeds=seeds, **kw)


def main():
    check = Check()
    C = "erase"                     # three epochs of erase, the best ALLaVA erase schedule

    print("Table 14, relative MAT difference at T=0 (%), all seeds and the common seeds 0 and 1")
    rows = [("ALR + IRA, 1 epoch vs. erase", dict(method="alr_ira", epochs=1), dict(method=C), "+5.33", "+5.24"),
            ("ALR + IRA, 3 epochs vs. erase", dict(method="alr_ira"), dict(method=C), "+8.54", "+8.64"),
            ("ALR, R=6 vs. R=14", dict(method="alr", depth=6), dict(method="alr"), "-0.37", "-0.31"),
            ("ALR, R=4 vs. R=14", dict(method="alr", depth=4), dict(method="alr"), "-0.93", "-0.81"),
            ("ALR, R=2 vs. R=14", dict(method="alr", depth=2), dict(method="alr"), "-2.68", "-2.41")]
    for label, a, b, p_all, p_common in rows:
        g_all = check(f"{label} all", gain(avg(runs(**a)), avg(runs(**b))), p_all)
        g_com = check(f"{label} common", gain(avg(runs(seeds=(0, 1), **a)), avg(runs(seeds=(0, 1), **b))), p_common)
        print(f"  {label:32s} {g_all:>7s} {g_com:>7s}")

    print("\nTable 15, relative MAT difference from erase (%)")
    o = [gain(avg(runs("alr", order=k)), avg(runs(C, order=k))) for k in (1, 2)]
    t1 = avg(runs(C, temperature=1))
    table15 = [("ALR, data-order replicate 1", o[0], "+3.99"),
               ("ALR, data-order replicate 2", o[1], "+4.37"),
               ("ALR, mean of the two orders", sum(o) / 2, "+4.18"),
               ("ALR, sampling at T=1", gain(avg(runs("alr", temperature=1)), t1), "+3.23"),
               ("ALR + IRA, sampling at T=1", gain(avg(runs("alr_ira", temperature=1)), t1), "+7.72"),
               ("ALR + IRA, chain decoding at T=0",
                gain(avg(runs("alr_ira", decoder="chain")), avg(runs(C, decoder="chain"))), "+5.90"),
               ("ALR + IRA against the slot-matched control",
                gain(avg(runs("alr_ira")), avg(runs("slot_control"))), "+3.92")]
    for label, value, printed in table15:
        print(f"  {label:44s} {check(label, value, printed):>7s}")

    print("\nFigure 4a, slot-matched control at T=0, MAT mean and SD")
    for label, method in (("ALR", "alr"), ("Slot control", "slot_control"), ("ALR + IRA", "alr_ira")):
        c = vision_cell(runs(method))
        print(f"  {label:12s} {c['n']} seeds {pm(c['avg'])}")
    g = gain(avg(runs("alr_ira", seeds=(0, 1))), avg(runs("slot_control")))
    print(f"  ALR + IRA over the control on the matched seeds 0 and 1  {check('figure 4a', g, '4.10')}%")

    print("\nAppendix G, erase comparator at T=0, relative to the soft gate")
    for label, method, printed in (("hard gate", "erase_hard", "-0.62"), ("no gate (KD)", "kd", "-4.85")):
        print(f"  {label:14s} {check(label, gain(avg(runs(method)), avg(runs(C))), printed):>7s}%")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
