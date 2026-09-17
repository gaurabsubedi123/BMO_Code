#!/usr/bin/env python3
"""Build .ipynb files from lightweight source files.

Cells are separated by marker lines:
    #%% md    -> markdown cell
    #%% code  -> code cell
This keeps notebooks authorable and diffable as plain text while the committed
artifact stays a real notebook.
"""
import json, sys, os

def parse(path):
    cells, state = [], {"kind": None, "buf": []}
    def flush():
        if state["kind"] is None:
            return
        src = "\n".join(state["buf"]).strip("\n")
        if not src.strip():
            return
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if state["kind"] == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": lines})
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.startswith("#%% "):
            flush()
            state = {"kind": line[4:].strip(), "buf": []}
        else:
            state["buf"].append(line)
    flush()
    return cells

def build(src, outdir):
    cells = parse(src)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(outdir, os.path.basename(src).replace(".py", ".ipynb"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"built {os.path.basename(out):46s} {len(cells):3d} cells")

if __name__ == "__main__":
    outdir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1])))
    for s in sys.argv[1:]:
        build(s, outdir)
