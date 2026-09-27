"""Table 8: vision-language tau by domain after three epochs, mean and SD across training seeds."""
from common import DOMAINS, NAMES, Check, find, pm, vision_cell

METHODS = ("dflash", "erase", "alr", "alr_ira")
PANELS = [("8b", "allava", 0, "Qwen3-VL-8B trained on ALLaVA, T=0"),
          ("8b", "allava", 1, "Qwen3-VL-8B trained on ALLaVA, T=1"),
          ("8b", "sharegpt4v", 0, "Qwen3-VL-8B trained on ShareGPT4V, T=0"),
          ("8b", "sharegpt4v", 1, "Qwen3-VL-8B trained on ShareGPT4V, T=1"),
          ("4b", "allava", 0, "Qwen3-VL-4B trained on ALLaVA, T=0"),
          ("4b", "allava", 1, "Qwen3-VL-4B trained on ALLaVA, T=1")]
# COCO, TextVQA, DocVQA, Avg. per method, as printed
PAPER = """
2.563±0.006 2.655±0.010 3.501±0.010 2.878±0.007
3.124±0.006 3.375±0.005 4.204±0.037 3.539±0.010
3.242±0.007 3.655±0.024 4.261±0.016 3.696±0.005
3.376±0.007 3.851±0.011 4.361±0.065 3.841±0.021
2.435±0.009 2.601±0.013 3.436±0.005 2.792±0.002
2.952±0.007 3.196±0.012 4.115±0.021 3.386±0.009
3.015±0.012 3.422±0.024 4.140±0.026 3.495±0.007
3.150±0.006 3.615±0.015 4.261±0.026 3.647±0.005
2.542±0.009 2.691±0.004 3.216±0.028 2.802±0.003
3.403±0.010 3.326±0.023 4.023±0.036 3.571±0.016
3.614±0.003 3.398±0.009 4.035±0.017 3.673±0.007
3.817±0.024 3.600±0.024 4.073±0.040 3.825±0.029
2.395±0.003 2.599±0.007 3.183±0.000 2.706±0.003
3.132±0.000 3.207±0.031 3.874±0.014 3.388±0.015
3.306±0.020 3.249±0.004 3.910±0.021 3.476±0.015
3.478±0.004 3.379±0.019 3.975±0.030 3.602±0.017
2.619±0.005 2.704±0.008 3.529±0.003 2.924±0.006
3.185±0.007 3.347±0.016 4.215±0.044 3.555±0.004
3.259±0.002 3.550±0.025 4.124±0.051 3.627±0.024
3.392±0.002 3.785±0.017 4.303±0.019 3.809±0.011
2.476±0.000 2.554±0.001 3.383±0.007 2.776±0.002
2.947±0.004 3.105±0.024 4.046±0.014 3.333±0.011
3.010±0.011 3.287±0.042 4.008±0.047 3.410±0.024
3.105±0.005 3.440±0.018 4.135±0.024 3.535±0.002
""".split("\n")[1:-1]


def main():
    check = Check()
    rows = iter(PAPER)
    for target, corpus, t, title in PANELS:
        print(f"\n{title}")
        print(f"{'method':10s} {'n':>2s} " + " ".join(f"{d:>12s}" for d in DOMAINS + ("avg",)))
        for m in METHODS:
            cell = vision_cell(find("acceptance", target, corpus, m, temperature=t))
            printed = next(rows).split()
            out = []
            for key, p in zip(DOMAINS + ("avg",), printed):
                mean, sd = p.split("±")
                check(f"{title} {m} {key} mean", cell[key][0], mean)
                check(f"{title} {m} {key} sd", cell[key][1], sd)
                out.append(pm(cell[key]))
            print(f"{NAMES[m]:10s} {cell['n']:2d} " + " ".join(f"{x:>12s}" for x in out))
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
