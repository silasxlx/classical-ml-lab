from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.mark.notebook
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_is_clean_and_executes_from_an_empty_kernel(path: Path, tmp_path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
    assert "classical_ml_lab" in code
    assert ".fit(" not in code
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    executed = client.execute()
    assert all(
        output.output_type != "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
