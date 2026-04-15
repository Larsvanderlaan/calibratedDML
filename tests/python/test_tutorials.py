from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = ROOT / "Python" / "tutorials"

SCRIPT_CASES = [
    "standard_workflows.py",
    "custom_learners.py",
    "supplied_nuisances.py",
    "adaptive_binary_treatment.py",
]

NOTEBOOK_CASES = [
    "standard-workflows.ipynb",
    "custom-learners.ipynb",
    "supplied-nuisances.ipynb",
    "adaptive-binary-treatment.ipynb",
]


@pytest.mark.parametrize("script_name", SCRIPT_CASES)
def test_tutorial_script_runs(script_name):
    completed = subprocess.run(
        [sys.executable, str(TUTORIAL_DIR / script_name)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tutorial completed" in completed.stdout.lower()


@pytest.mark.parametrize("notebook_name", NOTEBOOK_CASES)
def test_tutorial_notebook_executes(notebook_name):
    notebook_path = TUTORIAL_DIR / notebook_name
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(TUTORIAL_DIR)}},
    )
    try:
        executed = client.execute()
    except PermissionError as exc:
        pytest.skip(f"Notebook execution is blocked in this local sandbox: {exc}")
    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            assert output.output_type != "error"
