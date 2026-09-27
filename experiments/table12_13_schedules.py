"""Tables 12 and 13, and Table 3: tau under other training schedules and ALR rollout depths, T=0.

Table 3 is the Avg. column of the three-epoch rows of Table 13.
"""
from common import DOMAINS, Check, find, pm, vision_cell

# (label, corpus, method, epochs, depth, gamma), then COCO, TextVQA, DocVQA, Avg. as printed
TABLE12 = [
    ("Erase, 1 epoch", "allava", "erase", 1, None, None, "3.106±0.014 3.381±0.028 4.197±0.036 3.533±0.006"),
    ("ALR, 1 epoch", "allava", "alr", 1, 14, None, "3.199±0.006 3.593±0.027 4.255±0.020 3.657±0.017"),
    ("ALR + IRA, 1 epoch", "allava", "alr_ira", 1, 14, None, "3.263±0.011 3.732±0.006 4.254±0.030 3.728±0.005"),
    ("Erase, 3 epochs", "allava", "erase", 3, None, None, "3.124±0.006 3.375±0.005 4.204±0.037 3.539±0.010"),
    ("Erase (gamma=0.5), 3 epochs", "allava", "erase", 3, None, 0.5,
     "3.090±0.009 3.341±0.004 4.115±0.016 3.489±0.006"),
    ("ALR, 3 epochs", "allava", "alr", 3, 14, None, "3.242±0.007 3.655±0.024 4.261±0.016 3.696±0.005"),
    ("ALR + IRA, 3 epochs", "allava", "alr_ira", 3, 14, None, "3.376±0.007 3.851±0.011 4.361±0.065 3.841±0.021"),
    ("Erase, 6 epochs", "allava", "erase", 6, None, None, "3.083±0.010 3.283±0.003 4.134±0.034 3.472±0.014"),
    ("Erase, 0.5 epoch", "sharegpt4v", "erase", 0.5, None, None, "3.302±0.009 3.408±0.014 4.151±0.029 3.601±0.017"),
    ("Erase, 1 epoch", "sharegpt4v", "erase", 1, None, None, "3.362±0.018 3.409±0.004 4.208±0.013 3.640±0.002"),
    ("ALR + IRA, 1 epoch", "sharegpt4v", "alr_ira", 1, 14, None, "3.658±0.012 3.542±0.032 4.163±0.014 3.778±0.003"),
    ("Erase, 3 epochs", "sharegpt4v", "erase", 3, None, None, "3.403±0.010 3.326±0.023 4.023±0.036 3.571±0.016"),
    ("ALR, 3 epochs", "sharegpt4v", "alr", 3, 14, None, "3.614±0.003 3.398±0.009 4.035±0.017 3.673±0.007"),
    ("ALR + IRA, 3 epochs", "sharegpt4v", "alr_ira", 3, 14, None, "3.817±0.024 3.600±0.024 4.073±0.040 3.825±0.029"),
]
TABLE13 = [
    ("ALLaVA, one epoch, R=14", "allava", "alr", 1, 14, None, "3.199±0.006 3.593±0.027 4.255±0.020 3.657±0.017"),
    ("ALLaVA, one epoch, R=6", "allava", "alr", 1, 6, None, "3.194±0.009 3.587±0.022 4.250±0.019 3.652±0.001"),
    ("ALLaVA, one epoch, R=4", "allava", "alr", 1, 4, None, "3.195±0.006 3.591±0.028 4.236±0.018 3.649±0.014"),
    ("ALLaVA, three epochs, R=14", "allava", "alr", 3, 14, None, "3.242±0.007 3.655±0.024 4.261±0.016 3.696±0.005"),
    ("ALLaVA, three epochs, R=6", "allava", "alr", 3, 6, None, "3.230±0.010 3.629±0.018 4.258±0.008 3.682±0.004"),
    ("ALLaVA, three epochs, R=4", "allava", "alr", 3, 4, None, "3.232±0.016 3.606±0.039 4.211±0.063 3.661±0.023"),
    ("ALLaVA, three epochs, R=2", "allava", "alr", 3, 2, None, "3.205±0.002 3.489±0.026 4.161±0.032 3.597±0.017"),
    ("ShareGPT4V, one epoch, R=4", "sharegpt4v", "alr", 1, 4, None,
     "3.524±0.005 3.460±0.020 4.139±0.019 3.695±0.011"),
    ("ShareGPT4V, three epochs, R=14", "sharegpt4v", "alr", 3, 14, None,
     "3.614±0.003 3.398±0.009 4.035±0.017 3.673±0.007"),
    ("ShareGPT4V, three epochs, R=4", "sharegpt4v", "alr", 3, 4, None,
     "3.606±0.005 3.384±0.019 4.016±0.019 3.659±0.014"),
]


def show(title, rows, check):
    print(f"\n{title}")
    print(f"{'setting':34s} {'n':>2s} " + " ".join(f"{d:>12s}" for d in DOMAINS + ("avg",)))
    for label, corpus, method, epochs, depth, gamma, printed in rows:
        cell = vision_cell(find("acceptance", "8b", corpus, method, epochs=epochs, depth=depth, gamma=gamma))
        for key, p in zip(DOMAINS + ("avg",), printed.split()):
            mean, sd = p.split("±")
            check(f"{title} {label} {key}", cell[key][0], mean)
            check(f"{title} {label} {key} sd", cell[key][1], sd)
        name = label if corpus == "allava" or title.startswith("Table 13") else label + ", ShareGPT4V"
        print(f"{name:34s} {cell['n']:2d} " + " ".join(f"{pm(cell[k]):>12s}" for k in DOMAINS + ("avg",)))


def main():
    check = Check()
    show("Table 12, epochs and envelope, T=0 (ALLaVA rows first)", TABLE12, check)
    show("Table 13, ALR rollout depth R, T=0", TABLE13, check)
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
