"""Analysis and verification code must not import litellm.

litellm fetches a remote price map when it is imported, so every analysis
script that pulled it in would reach the network and slow down. Each module
is imported in a fresh interpreter, where nothing else has loaded it yet.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "experiments.exclusions",
    "experiments.analysis.runs",
    "experiments.analysis.significance",
    "experiments.analysis.hop_level",
    "experiments.analysis.refusal_analysis",
    "experiments.verify",
    "scripts.verify_campaign",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_does_not_import_litellm(module):
    code = (
        "import sys, importlib; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        f"importlib.import_module({module!r}); "
        "print(sorted(m for m in ('litellm', 'dspy') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout + out.stderr
