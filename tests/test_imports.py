"""Every module must import on its own (guards against circular imports that only
work thanks to a lucky import order elsewhere)."""

import pkgutil
import subprocess
import sys

import pytest

import readback

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(readback.__path__, "readback.")
    if not m.name.endswith((".local", ".openai_compat"))  # optional extras
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_isolation(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
