"""The `alr-ira` command. `python -m alr_ira` runs the same thing."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from . import __version__


def _demo(args):
    from .toy import format_results, run_toy

    print(format_results(run_toy(worlds=args.worlds, seed=args.seed)))
    return 0


def _score(args):
    from .metrics import geomean, mat, speedup

    where = dict(w.split("=", 1) for w in args.where)
    groups = defaultdict(lambda: defaultdict(list))
    with open(args.outcomes, newline="") as fh:
        for row in csv.DictReader(fh):
            if any(row.get(k) != v for k, v in where.items()):
                continue
            g = groups[row[args.group]]
            for col in ("committed", "rounds", "ar_ms", "draft_ms"):
                if row.get(col, "") != "":
                    g[col].append(float(row[col]))
    if not groups:
        raise SystemExit("no rows match")
    timed = all(len(g["ar_ms"]) == len(g["committed"]) > 0 for g in groups.values())
    print(f"{args.group:16s} {'prompts':>7s} {'tau':>7s}" + (f" {'speedup':>8s}" if timed else ""))
    taus, speeds = [], []
    for name, g in groups.items():
        taus.append(mat(g["committed"], g["rounds"]))
        line = f"{name:16s} {len(g['committed']):7d} {taus[-1]:7.3f}"
        if timed:
            speeds.append(speedup(g["ar_ms"], g["draft_ms"]))
            line += f" {speeds[-1]:7.2f}x"
        print(line)
    if len(groups) > 1:
        line = f"{'geometric mean':16s} {'':7s} {geomean(taus):7.3f}"
        print(line + (f" {geomean(speeds):7.2f}x" if timed else ""))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="alr-ira", description="Target-rollout labels and in-rollout anchors for block drafters")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart on toy Markov targets")
    demo.add_argument("--worlds", type=int, default=5, help="toy worlds to average over")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_demo)

    score = sub.add_parser("score", help="accepted length and speedup from prompt-level outcomes")
    score.add_argument("outcomes", help="csv with columns committed and rounds, and optionally ar_ms and draft_ms")
    score.add_argument("--group", default="domain", help="column to report by, such as domain or task")
    score.add_argument("--where", nargs="*", default=[], metavar="COL=VALUE", help="keep only matching rows")
    score.set_defaults(func=_score)

    args = parser.parse_args(argv)
    return args.func(args)
