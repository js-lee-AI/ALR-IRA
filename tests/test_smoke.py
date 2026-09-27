import re
import subprocess
import sys
from pathlib import Path

import alr_ira

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, check=True,
                          timeout=120, cwd=ROOT).stdout


def test_import_does_not_pull_torch():
    # A fresh interpreter, since pytest plugins may have imported torch already.
    out = run("-c", "import sys, alr_ira; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "ALR + IRA    8.67    3200      1600" in out
    assert "on average: 7.05" in out


def test_cli_help_and_demo():
    assert "demo" in run("-m", "alr_ira", "--help")
    assert "ALR + IRA" in run("-m", "alr_ira", "demo", "--worlds", "1")


def test_cli_scores_a_timing_run():
    out = run("-m", "alr_ira", "score", "data/vision/timing.csv",
              "--where", "run=session1-8b-allava-alr_ira-e3-s0-t0-run1")
    assert "geometric mean             3.838    2.85x" in out


def test_readme_python_blocks_print_what_they_show():
    # the comment lines right after a print() are the output the README shows
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert len(blocks) == 3
    for block in blocks:
        shown, after_print = [], False
        for line in block.splitlines():
            if after_print and line.startswith("# "):
                shown.append(line[2:])
                continue
            after_print = line.startswith("print(")
        assert shown and run("-c", block).splitlines() == shown


def test_version():
    assert alr_ira.__version__ == "0.1.0"
