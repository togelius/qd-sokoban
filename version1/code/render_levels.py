"""Render selected example levels as figure PDFs for the paper."""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sokoban import Level


def _level_from_render(text: str) -> Level:
    return Level.parse(text)


# Color palette (boring, paper-friendly)
COLOR_WALL = "#3a3a3a"
COLOR_FLOOR = "#f4f1ea"
COLOR_GOAL = "#ffe4a1"
COLOR_BOX = "#b46a23"
COLOR_BOX_ON_GOAL = "#3f7a3f"
COLOR_PLAYER = "#1f78b4"


def draw_level(ax, lvl: Level, title: str = ""):
    H, W = lvl.H, lvl.W
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # invert y so row 0 is on top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    initial_state = lvl.initial_state()
    for r in range(H):
        for c in range(W):
            p = (r, c)
            if lvl.walls[r, c]:
                ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                facecolor=COLOR_WALL, edgecolor="none"))
                continue
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor=COLOR_FLOOR, edgecolor="#cfcabb",
                                            linewidth=0.4))
            if p in lvl.goals:
                ax.add_patch(mpatches.Circle((c, r), 0.22, facecolor=COLOR_GOAL,
                                             edgecolor="#b78a3a", linewidth=0.8))
            if p in initial_state.boxes:
                fc = COLOR_BOX_ON_GOAL if p in lvl.goals else COLOR_BOX
                ax.add_patch(mpatches.Rectangle((c - 0.32, r - 0.32), 0.64, 0.64,
                                                facecolor=fc, edgecolor="#000000",
                                                linewidth=0.6))
            if p == initial_state.player:
                ax.add_patch(mpatches.Circle((c, r), 0.28, facecolor=COLOR_PLAYER,
                                             edgecolor="#000000", linewidth=0.6))
    if title:
        ax.set_title(title, fontsize=8)


def figure_grid(levels: List[Tuple[Level, str]], n_cols: int, out_path: str,
                cell_size: float = 1.2):
    n = len(levels)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_size, n_rows * cell_size))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    for i, (lvl, title) in enumerate(levels):
        r, c = i // n_cols, i % n_cols
        draw_level(axes[r][c], lvl, title)
    # Blank out unused
    for i in range(len(levels), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r][c].axis("off")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(results_path: str, out_dir: str, per_cond: int = 6):
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # Group by condition; collect cells from all seeds
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        for c in r["common"]["cells"]:
            by_cond[r["cond"]].append(c)

    cond_order = ["RAND", "STR", "PLAY", "SKILL"]
    cond_pretty = {"RAND": "Random", "STR": "ME-STR",
                   "PLAY": "ME-PLAY", "SKILL": "ME-SKILL"}

    # Per-condition: pick best-fitness per common cell, then take top per_cond
    # spread across the cells.
    for cond in cond_order:
        cells = by_cond[cond]
        by_idx = {}
        for c in cells:
            idx = tuple(c["idx"])
            if idx not in by_idx or c["fit"] > by_idx[idx]["fit"]:
                by_idx[idx] = c
        # Sort by fitness desc and take top per_cond
        ranked = sorted(by_idx.values(), key=lambda c: -c["fit"])[:per_cond]
        rendered = []
        for c in ranked:
            lvl = _level_from_render(c["render"])
            t = (f"L={c['plan_len']}, P={c['n_pushes']}, "
                 f"B={c['n_boxes']}, f={c['fit']:.2f}")
            rendered.append((lvl, t))
        out_path = os.path.join(out_dir, f"examples_{cond}.pdf")
        figure_grid(rendered, n_cols=per_cond, out_path=out_path)
        print(f"  {cond}: {len(rendered)} examples -> {out_path}")

    # Combined: 1 row per condition, top-4 examples each
    rows = []
    for cond in cond_order:
        cells = by_cond[cond]
        by_idx = {}
        for c in cells:
            idx = tuple(c["idx"])
            if idx not in by_idx or c["fit"] > by_idx[idx]["fit"]:
                by_idx[idx] = c
        ranked = sorted(by_idx.values(), key=lambda c: -c["fit"])[:4]
        rows.append((cond, ranked))

    n_cols = 4
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.3 + 0.4, n_rows * 1.4))
    for ri, (cond, cells) in enumerate(rows):
        for ci, c in enumerate(cells):
            lvl = _level_from_render(c["render"])
            t = f"L={c['plan_len']}  P={c['n_pushes']}  B={c['n_boxes']}"
            draw_level(axes[ri][ci], lvl, t)
        for ci in range(len(cells), n_cols):
            axes[ri][ci].axis("off")
        # row label
        axes[ri][0].text(-1.0, (axes[ri][0].get_ylim()[0] + axes[ri][0].get_ylim()[1]) / 2,
                         cond_pretty[cond], rotation=90,
                         ha="center", va="center", fontsize=10)
    plt.tight_layout(pad=0.4)
    out_path = os.path.join(out_dir, "examples_combined.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  combined -> {out_path}")


if __name__ == "__main__":
    main(
        results_path=os.path.join(os.path.dirname(__file__), "..", "results", "raw_results.pkl"),
        out_dir=os.path.join(os.path.dirname(__file__), "..", "results", "figures"),
    )
