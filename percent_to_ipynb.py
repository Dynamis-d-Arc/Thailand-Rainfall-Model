"""Convert a jupytext-style percent-format .py into a .ipynb.

The notebooks in this project are authored as plain .py files in percent format and converted
here, so the source stays diffable and the notebook stays the artifact. An earlier copy of this
helper lived only in %TEMP% and was lost when Windows cleared it; this one lives in the project.

Cell markers:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (leading "# " is stripped from each line)

Usage:
    python percent_to_ipynb.py source.py [output.ipynb]
"""

import json
import sys
from pathlib import Path


def split_cells(text):
    """Yield (kind, source) pairs. Anything before the first marker becomes a leading code cell."""
    cells, kind, buffer = [], "code", []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buffer:
                cells.append((kind, buffer))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        cells.append((kind, buffer))
    return cells


def clean(kind, lines):
    """Strip markdown comment prefixes and surrounding blank lines."""
    if kind == "markdown":
        out = []
        for line in lines:
            if line.startswith("# "):
                out.append(line[2:])
            elif line.strip() == "#":
                out.append("")
            else:
                out.append(line)
        lines = out
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_notebook(text):
    cells = []
    for kind, lines in split_cells(text):
        lines = clean(kind, lines)
        if not lines:
            continue
        source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": source})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix(".ipynb")
    notebook = to_notebook(source.read_text(encoding="utf-8"))
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    kinds = [c["cell_type"] for c in notebook["cells"]]
    print(f"{source} -> {target}: {len(kinds)} cells "
          f"({kinds.count('code')} code, {kinds.count('markdown')} markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
