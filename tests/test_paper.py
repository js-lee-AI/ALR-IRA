"""Every script in experiments/ rebuilds its paper values from data/ and compares them with the PDF."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "table1_main.py": 176, "table2_one_epoch.py": 19, "table4_regeneration.py": 17,
    "table5_label_source.py": 16, "table8_domains.py": 192, "table9_10_text.py": 224,
    "table11_repeated_timing.py": 10, "table12_13_schedules.py": 192, "table14_15_controls.py": 20,
    "table16_17_speed.py": 72, "figure2_gains.py": 7, "appendix_e_resampling.py": 10,
}


@pytest.mark.parametrize("script, count", sorted(SCRIPTS.items()))
def test_script_reproduces_the_printed_values(script, count):
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / script)], capture_output=True,
                         text=True, timeout=300, cwd=ROOT / "experiments")
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"{count} of {count} printed values reproduced" in out.stdout


def test_every_script_is_listed():
    shipped = {p.name for p in (ROOT / "experiments").glob("*.py")} - {"common.py"}
    assert shipped == set(SCRIPTS)
