"""Render top-fitness elites per condition to a small grid figure.

Outputs:
    ../results/sokoban/figures/examples_<COND>.pdf
    ../results/platformer/figures/examples_<COND>.pdf
    ../results/sokoban/figures/examples_combined.pdf
    ../results/platformer/figures/examples_combined.pdf
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["RAND", "STR", "PLAY", "SKILL", "COMMON"]


def load_runs(outdir):
    with open(os.path.join(outdir, "raw_runs.pkl"), "rb") as f:
        return pickle.load(f)


def top_elites_per_cond(runs, n_each=6) -> Dict[str, list]:
    """For each condition, gather best-fitness elites (across seeds) up to n_each."""
    out = defaultdict(list)
    for r in runs:
        cells = r["common_archive"]["cells"]
        for c in cells:
            out[r["cond"]].append((c["fitness"], c["render"], c))
    for c in out:
        out[c] = sorted(out[c], key=lambda x: -x[0])[:n_each]
    return out


# ---------------------------------------------------------------------------
# Sokoban: render ASCII grid with colors
# ---------------------------------------------------------------------------

SOKO_COLORS = {
    "#": (0.15, 0.15, 0.15),    # wall
    " ": (0.95, 0.95, 0.95),    # floor
    ".": (0.85, 0.95, 0.85),    # goal
    "$": (0.95, 0.75, 0.40),    # box
    "*": (0.50, 0.85, 0.50),    # box on goal
    "@": (0.55, 0.70, 0.95),    # player
    "+": (0.30, 0.55, 0.90),    # player on goal
}


def draw_sokoban(ax, render: str):
    rows = render.splitlines()
    H = len(rows)
    W = max(len(r) for r in rows)
    grid = np.ones((H, W, 3))
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            grid[r, c] = SOKO_COLORS.get(ch, (1, 1, 1))
    ax.imshow(grid, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Platformer: render tile grid
# ---------------------------------------------------------------------------

PLAT_COLORS = {
    ".": (0.95, 0.95, 0.97),    # empty (sky)
    "X": (0.20, 0.50, 0.25),    # solid (ground)
    "S": (0.80, 0.20, 0.20),    # spike
}


def draw_platformer(ax, render: str):
    rows = render.splitlines()
    H = len(rows)
    W = max(len(r) for r in rows)
    grid = np.ones((H, W, 3))
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            grid[r, c] = PLAT_COLORS.get(ch, (1, 1, 1))
    ax.imshow(grid, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(0.5)


def make_combined_figure(runs, outpath, domain: str, n_each: int = 4):
    """Grid: rows = conditions, cols = top n_each elites by fitness."""
    top = top_elites_per_cond(runs, n_each=n_each)
    conds = [c for c in CONDITIONS if c in top]
    rows = len(conds)
    cols = n_each
    if domain == "sokoban":
        figsize = (cols * 1.0, rows * 1.0)
        draw = draw_sokoban
    else:
        figsize = (cols * 1.8, rows * 0.8)
        draw = draw_platformer
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, cond in enumerate(conds):
        for j in range(cols):
            ax = axes[i][j]
            if j < len(top[cond]):
                fit, render, meta = top[cond][j]
                draw(ax, render)
                ax.set_title(f"f={fit:.2f}", fontsize=7, pad=2)
            else:
                ax.axis("off")
        axes[i][0].set_ylabel(cond, fontsize=9, rotation=0,
                              labelpad=18, ha="right", va="center")
    fig.suptitle(f"Top elites by fitness — {domain}", fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["sokoban", "platformer"])
    args = ap.parse_args()
    outdir = f"../results/{args.domain}"
    runs = load_runs(outdir)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    make_combined_figure(runs, os.path.join(figdir, "examples_combined.pdf"),
                         args.domain)
    print(f"wrote {figdir}/examples_combined.pdf")


if __name__ == "__main__":
    main()
