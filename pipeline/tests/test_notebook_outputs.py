"""Every tracked notebook must be committed with its cell outputs cleared.

Stdlib ``unittest`` only, matching the rest of the suite. The checker is a pure
function over a parsed notebook dict, so it is tested without git; the
repository test then applies it to every ``.ipynb`` that ``git ls-files`` lists
and is skipped, with a reason, outside a git checkout.

Why this is enforced: executing a notebook stores each cell's output inside the
``.ipynb``, and those outputs print real dataset paths. See
``test_dt4ag_config.py`` for the term check on the committed notebook.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def find_dirty_cells(notebook):
    """Return a list of problems for a parsed notebook (a dict).

    Each problem is ``(cell_index, reason)``. A code cell is dirty when it has
    a non-empty ``outputs`` list or an ``execution_count`` that is not null.
    Markdown and raw cells are ignored. An empty list means the notebook is
    clean.
    """
    problems = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs") or []
        if outputs:
            problems.append((index, f"{len(outputs)} stored output(s)"))
        count = cell.get("execution_count")
        if count is not None:
            problems.append((index, f"execution_count is {count!r}, not null"))
    return problems


def describe_problems(path, problems):
    """Build the failure message: file, cell indexes, and how to fix it."""
    lines = [f"{path} was committed with cell outputs or execution counts:"]
    lines.extend(f"  cell {index}: {reason}" for index, reason in problems)
    lines.append(
        f"Clear them before committing: nbstripout {path}")
    return "\n".join(lines)


def tracked_notebooks():
    """Return ``(repo_root, [relative paths])`` or raise ``unittest.SkipTest``."""
    if shutil.which("git") is None:
        raise unittest.SkipTest("git is not installed; cannot list notebooks")
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=TESTS_DIR,
            capture_output=True, text=True, check=True).stdout.strip()
        listed = subprocess.run(
            ["git", "ls-files", "-z", "*.ipynb"], cwd=top,
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        raise unittest.SkipTest(
            f"not inside a git checkout, cannot list tracked notebooks: {exc}")
    return Path(top), [p for p in listed.split("\0") if p]


def code_cell(source, outputs=None, execution_count=None):
    return {"cell_type": "code", "metadata": {}, "source": [source],
            "outputs": [] if outputs is None else outputs,
            "execution_count": execution_count}


class TestFindDirtyCells(unittest.TestCase):
    """The checker itself, on in-memory notebooks."""

    def test_clean_notebook_passes_and_dirty_one_is_flagged(self):
        clean = {"cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
            code_cell("print('x')"),
        ]}
        dirty = {"cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
            code_cell("print('x')"),
            code_cell("print('y')", execution_count=3, outputs=[
                {"output_type": "stream", "name": "stdout",
                 "text": ["/data/some/path\n"]}]),
        ]}
        self.assertEqual(find_dirty_cells(clean), [])
        problems = find_dirty_cells(dirty)
        self.assertEqual([index for index, _ in problems], [2, 2])
        message = describe_problems("nb/example.ipynb", problems)
        self.assertIn("nb/example.ipynb", message)
        self.assertIn("cell 2", message)
        self.assertIn("nbstripout nb/example.ipynb", message)

    def test_execution_count_alone_is_flagged(self):
        notebook = {"cells": [code_cell("1 + 1", execution_count=1)]}
        self.assertEqual([i for i, _ in find_dirty_cells(notebook)], [0])


class TestTrackedNotebooksAreClean(unittest.TestCase):
    def test_every_tracked_notebook_has_outputs_cleared(self):
        root, notebooks = tracked_notebooks()
        if not notebooks:
            self.skipTest("no tracked .ipynb files")
        for relative in notebooks:
            with self.subTest(notebook=relative):
                path = root / relative
                if not path.is_file():
                    continue  # tracked but deleted in this working tree
                data = json.loads(path.read_text(encoding="utf-8"))
                problems = find_dirty_cells(data)
                self.assertEqual(
                    problems, [], describe_problems(relative, problems))


if __name__ == "__main__":
    unittest.main()
