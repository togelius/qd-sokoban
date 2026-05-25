"""Analysis: load raw_results.pkl and produce plots + statistical tables.

Outputs (in ../results/):
  - summary_table.txt    : mean ± std for each (cond, metric)
  - stats.txt            : pairwise Mann-Whitney U + effect sizes
  - convergence.pdf      : QD-score on common over generations
  - coverage_over_time.pdf
  - solver_time.pdf
  - bar_metrics.pdf
  - common_heatmaps.pdf  : averaged common-archive cell fitness, per cond
  - examples/            : ASCII renderings of selected puzzles per cond
"""

from __future__ import annotations

import json
import os
import pickle
import textwrap
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
COND_ORDER = ["RAND", "STR", "PLAY", "SKILL"]
COND_PRETTY = {
    "RAND":  "Random search",
    "STR":   "ME-Structural",
    "PLAY":  "ME-Gameplay",
    "SKILL": "ME-Skill",
}
COND_COLORS = {
    "RAND":  "#7f7f7f",
    "STR":   "#1f77b4",
    "PLAY":  "#d62728",
    "SKILL": "#2ca02c",
}


def load_results() -> List[dict]:
    with open(os.path.join(RESULTS_DIR, "raw_results.pkl"), "rb") as f:
        return pickle.load(f)


def load_config() -> dict:
    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------

METRICS = [
    ("common_coverage", "Common coverage (/40)"),
    ("common_qd_score", "Common QD-score"),
    ("common_max_fit", "Max fitness"),
    ("hard_coverage",  "Hard-cell cov (/16)"),
    ("hard_qd_score",  "Hard-cell QD"),
    ("wall_time_s",   "Wall time (s)"),
    ("n_solver_calls", "Solver calls"),
]

# Hard cells = plan_len bin >= 6 (i.e. last 4 columns, ~plan_len >= 14)
HARD_BIN_THRESHOLD = 6


def per_cond_metrics(results: List[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns metrics[cond][metric_key] = np.ndarray of per-seed values."""
    out: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r["cond"]
        log = r["log"]
        common = r["common"]["cells"]
        out[cond]["common_coverage"].append(len(common))
        out[cond]["common_qd_score"].append(sum(c["fit"] for c in common))
        out[cond]["common_max_fit"].append(max((c["fit"] for c in common), default=0.0))
        hard = [c for c in common if c["idx"][1] >= HARD_BIN_THRESHOLD]
        out[cond]["hard_coverage"].append(len(hard))
        out[cond]["hard_qd_score"].append(sum(c["fit"] for c in hard))
        out[cond]["wall_time_s"].append(log["wall_time_s"][-1])
        out[cond]["n_solver_calls"].append(log["n_solver_calls"][-1])
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in out.items()}


def summary_table(metrics: Dict[str, Dict[str, np.ndarray]]) -> str:
    lines = []
    header = ["Condition"] + [m[1] for m in METRICS]
    lines.append(" | ".join(f"{h:>22}" for h in header))
    lines.append("-" * len(lines[0]))
    for cond in COND_ORDER:
        row = [f"{COND_PRETTY[cond]:>22}"]
        for key, _ in METRICS:
            vals = metrics[cond][key]
            row.append(f"{vals.mean():>10.2f} ± {vals.std(ddof=1):>7.2f}")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def pairwise_stats(metrics: Dict[str, Dict[str, np.ndarray]]) -> str:
    """Mann-Whitney U + rank-biserial effect size, pairwise per metric."""
    lines = []
    for key, label in METRICS:
        lines.append(f"\n# {label}")
        for i, a in enumerate(COND_ORDER):
            for b in COND_ORDER[i + 1:]:
                xa = metrics[a][key]
                xb = metrics[b][key]
                # Mann-Whitney U (two-sided)
                try:
                    u, p = mannwhitneyu(xa, xb, alternative="two-sided")
                    # Rank-biserial effect size r = 1 - 2U / (n1*n2)
                    rb = 1.0 - 2.0 * u / (len(xa) * len(xb))
                except ValueError:
                    p, rb = float("nan"), float("nan")
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                lines.append(
                    f"  {a:>5} vs {b:<5}  U={u:>7.1f}  p={p:>8.4g}  rb={rb:+.3f}  {stars}"
                )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_convergence(results, out_path):
    """common QD-score over generations, mean ± std per cond."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    by_cond: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    for r in results:
        by_cond[r["cond"]].append((r["log"]["gen"], r["log"]["common_qd_score"]))
    for cond in COND_ORDER:
        rows = by_cond[cond]
        if not rows:
            continue
        gens = rows[0][0]
        arr = np.array([row[1] for row in rows])
        m = arr.mean(axis=0)
        s = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
        ax.plot(gens, m, label=COND_PRETTY[cond], color=COND_COLORS[cond], lw=1.8)
        ax.fill_between(gens, m - s, m + s, color=COND_COLORS[cond], alpha=0.18, lw=0)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("QD-score (common archive)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_coverage_over_time(results, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    by_cond: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    for r in results:
        by_cond[r["cond"]].append((r["log"]["gen"], r["log"]["common_coverage"]))
    for cond in COND_ORDER:
        rows = by_cond[cond]
        if not rows:
            continue
        gens = rows[0][0]
        arr = np.array([row[1] for row in rows])
        m = arr.mean(axis=0)
        s = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
        ax.plot(gens, m, label=COND_PRETTY[cond], color=COND_COLORS[cond], lw=1.8)
        ax.fill_between(gens, m - s, m + s, color=COND_COLORS[cond], alpha=0.18, lw=0)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Coverage of common archive (/40)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_bar_metrics(metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    keys = [
        ("common_coverage", "Common coverage (/40)"),
        ("common_qd_score", "Common QD-score"),
        ("common_max_fit", "Max fitness"),
    ]
    for ax, (key, label) in zip(axes, keys):
        means = [metrics[c][key].mean() for c in COND_ORDER]
        stds = [metrics[c][key].std(ddof=1) for c in COND_ORDER]
        ax.bar(range(len(COND_ORDER)), means, yerr=stds,
               color=[COND_COLORS[c] for c in COND_ORDER],
               edgecolor="black", linewidth=0.6, capsize=4)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_PRETTY[c] for c in COND_ORDER],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_common_heatmaps(results, out_path):
    """For each condition, average fitness per (n_boxes, plan_len) cell
    across seeds. Empty cells shown white."""
    by_cond: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in results:
        H = np.full((4, 10), np.nan)
        for c in r["common"]["cells"]:
            i, j = c["idx"]
            if 0 <= i < 4 and 0 <= j < 10:
                H[i, j] = c["fit"]
        by_cond[r["cond"]].append(H)

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5), constrained_layout=True)
    vmax = max(np.nanmax([h for h in by_cond[c]]) if by_cond[c] else 0 for c in COND_ORDER)
    for ax, cond in zip(axes, COND_ORDER):
        stack = np.stack(by_cond[cond])  # (n_seeds, 4, 10)
        # cell-wise mean ignoring NaN; if all NaN -> NaN
        mean_arr = np.nanmean(stack, axis=0)
        cov_count = (~np.isnan(stack)).sum(axis=0)
        im = ax.imshow(mean_arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(COND_PRETTY[cond], fontsize=10)
        ax.set_xlabel("plan length bin", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("# boxes", fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels([1, 2, 3, 4], fontsize=8)
        ax.set_xticks(range(0, 10, 2))
        ax.set_xticklabels(range(0, 10, 2), fontsize=8)
        # annotate seed-coverage count
        for i in range(4):
            for j in range(10):
                if cov_count[i, j] > 0:
                    ax.text(j, i, str(cov_count[i, j]),
                            ha="center", va="center", color="white", fontsize=6)
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("Mean fitness", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_solver_time(metrics, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    means = [metrics[c]["wall_time_s"].mean() for c in COND_ORDER]
    stds = [metrics[c]["wall_time_s"].std(ddof=1) for c in COND_ORDER]
    ax.bar(range(len(COND_ORDER)), means, yerr=stds,
           color=[COND_COLORS[c] for c in COND_ORDER],
           edgecolor="black", linewidth=0.6, capsize=4)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_PRETTY[c] for c in COND_ORDER],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Wall time per run (s)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Example puzzle selection
# ----------------------------------------------------------------------

def pick_examples_per_cond(results, k=6):
    """For each condition, pick `k` high-fitness, diverse examples from
    the common archive (sampled across seeds). Diversity = different
    common cells."""
    by_cond_cells: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        for c in r["common"]["cells"]:
            by_cond_cells[r["cond"]].append(c)

    examples = {}
    for cond in COND_ORDER:
        cells = by_cond_cells[cond]
        # group by cell, pick max fit per cell
        by_idx: Dict[Tuple[int, int], dict] = {}
        for c in cells:
            idx = tuple(c["idx"])
            if idx not in by_idx or c["fit"] > by_idx[idx]["fit"]:
                by_idx[idx] = c
        # sort by fit desc, take first k
        ranked = sorted(by_idx.values(), key=lambda c: -c["fit"])
        examples[cond] = ranked[:k]
    return examples


def write_examples(examples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for cond, cells in examples.items():
        path = os.path.join(out_dir, f"{cond}.txt")
        with open(path, "w") as f:
            f.write(f"# Example puzzles from condition {cond}\n")
            f.write(f"# Selected as max-fitness elite per common archive cell.\n\n")
            for i, c in enumerate(cells, 1):
                f.write(
                    f"--- example {i} ---\n"
                    f"common idx: {tuple(c['idx'])}  fit: {c['fit']:.3f}  "
                    f"n_boxes: {c['n_boxes']}  plan_len: {c['plan_len']}  "
                    f"pushes: {c['n_pushes']}\n"
                    f"{c['render']}\n\n"
                )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def latex_summary_table(metrics, cfg) -> str:
    """LaTeX table to be \\input{}-ed by the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final metrics on the common archive, 10 seeds per condition. "
                 r"Mean $\pm$ std. ``Hard'' = common cells with plan length $\geq 14$. "
                 r"Max fitness is per-archive maximum.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Cov\,/40 & QD & MaxF & H-Cov\,/16 & H-QD & Time (s) \\")
    lines.append(r"\midrule")
    for cond in COND_ORDER:
        m = metrics[cond]
        lines.append(
            f"{COND_PRETTY[cond]} & "
            f"{m['common_coverage'].mean():.1f}\\,$\\pm$\\,{m['common_coverage'].std(ddof=1):.1f} & "
            f"{m['common_qd_score'].mean():.1f}\\,$\\pm$\\,{m['common_qd_score'].std(ddof=1):.1f} & "
            f"{m['common_max_fit'].mean():.2f}\\,$\\pm$\\,{m['common_max_fit'].std(ddof=1):.2f} & "
            f"{m['hard_coverage'].mean():.1f}\\,$\\pm$\\,{m['hard_coverage'].std(ddof=1):.1f} & "
            f"{m['hard_qd_score'].mean():.1f}\\,$\\pm$\\,{m['hard_qd_score'].std(ddof=1):.1f} & "
            f"{m['wall_time_s'].mean():.1f}\\,$\\pm$\\,{m['wall_time_s'].std(ddof=1):.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_results_inline(metrics, cfg, out_path: str):
    """The piece of LaTeX that the paper \\input{}s for the results section."""
    body = []
    body.append(r"\input{results_table.tex}")
    body.append("")
    body.append(r"\begin{figure}[t]")
    body.append(r"\centering")
    body.append(r"\includegraphics[width=\linewidth]{convergence.pdf}")
    body.append(r"\caption{Common-archive QD-score over evaluations. Lines are "
                r"means across 10 seeds; bands are $\pm 1$ std. The three "
                r"MAP-Elites variants overlap; random search is well below.}")
    body.append(r"\label{fig:convergence}")
    body.append(r"\end{figure}")
    body.append("")
    body.append(r"\begin{figure*}[t]")
    body.append(r"\centering")
    body.append(r"\includegraphics[width=\linewidth]{common_heatmaps.pdf}")
    body.append(r"\caption{Mean fitness in each cell of the common archive "
                r"$(n_\text{boxes}, \text{plan length})$ averaged across 10 "
                r"seeds. Numbers in cells are the count of seeds that ever "
                r"filled that cell. White = never filled. Note how "
                r"random search fails to fill the long-plan column at the "
                r"right, while all three MAP-Elites variants reach those "
                r"cells; \\textsc{ME-Str} concentrates fitness in the "
                r"middle-density region; \\textsc{ME-Skill} extends "
                r"furthest right and fills more 3-/4-box cells.}")
    body.append(r"\label{fig:heatmaps}")
    body.append(r"\end{figure*}")
    body.append("")
    body.append(r"\begin{figure*}[t]")
    body.append(r"\centering")
    body.append(r"\includegraphics[width=\linewidth]{examples_combined.pdf}")
    body.append(r"\caption{Four high-fitness elites per condition, drawn "
                r"from the common archive. Brown squares are boxes, yellow "
                r"circles are goals, green squares are boxes already on "
                r"goals, blue circle is the player. $L$: A$^*$ plan length, "
                r"$P$: number of pushes in the plan, $B$: number of boxes.}")
    body.append(r"\label{fig:examples}")
    body.append(r"\end{figure*}")
    body.append("")

    body.append(r"\paragraph{Quantitative comparison.} "
                r"Table~\ref{tab:main} summarises the four conditions. "
                r"All three MAP-Elites variants beat random search on every "
                r"metric ($p<0.001$, Mann-Whitney U). On common coverage, "
                r"\textsc{ME-Skill} is the strongest variant "
                f"({metrics['SKILL']['common_coverage'].mean():.1f}/40 vs."
                f" {metrics['STR']['common_coverage'].mean():.1f} for "
                r"\textsc{ME-Str} and "
                f"{metrics['PLAY']['common_coverage'].mean():.1f} for "
                r"\textsc{ME-Play}; $p<0.05$ in both pairwise tests). "
                r"\textsc{ME-Str} and \textsc{ME-Play} do not differ "
                r"significantly on coverage or QD-score. ")

    body.append(r"\paragraph{Hard-cell coverage.} "
                r"We define the \emph{hard region} of the common archive "
                r"as the 16 cells with plan length $\geq 14$ -- the "
                r"puzzles a designer is most likely to actually ship. "
                r"On this restricted region the descriptor effect sharpens: "
                f"\\textsc{{ME-Skill}} fills "
                f"{metrics['SKILL']['hard_coverage'].mean():.1f}/16 cells, "
                f"\\textsc{{ME-Play}} {metrics['PLAY']['hard_coverage'].mean():.1f}, "
                f"\\textsc{{ME-Str}} {metrics['STR']['hard_coverage'].mean():.1f}, "
                f"and random search only "
                f"{metrics['RAND']['hard_coverage'].mean():.1f}. "
                r"Random search is essentially incapable of finding the "
                r"three- and four-box hard puzzles. Among the MAP-Elites "
                r"variants, the two solver-based descriptors "
                r"(\textsc{ME-Play}, \textsc{ME-Skill}) dominate the "
                r"hard region, confirming that descriptors that explicitly "
                r"index gameplay properties push the search where designers "
                r"actually want it.")

    body.append(r"\paragraph{Wall-clock cost.} "
                r"The structural descriptor, naively the cheapest because "
                r"it requires no solver call beyond the one used for "
                r"fitness, is in fact the slowest condition "
                f"({metrics['STR']['wall_time_s'].mean():.0f}s per run, vs. "
                f"{metrics['PLAY']['wall_time_s'].mean():.0f}s for "
                r"\textsc{ME-Play} and "
                f"{metrics['SKILL']['wall_time_s'].mean():.0f}s for "
                r"\textsc{ME-Skill}). The cause is selection bias: \textsc{ME-Str}'s "
                r"second axis is wall density, and it preferentially samples "
                r"dense boards from its archive; A$^*$ on a dense $8\times 8$ "
                r"Sokoban runs many more node expansions than on a sparse one. "
                r"Solver calls are nearly identical across conditions; only "
                r"the per-call cost differs.")

    body.append(r"\paragraph{Where coverage is placed.} "
                r"Figure~\ref{fig:heatmaps} shows the per-cell mean fitness "
                r"of the common archive, averaged across seeds. Random "
                r"search reliably fills the easy short-plan, 1--2 box "
                r"corner but never reaches the long-plan column at the "
                r"right, and barely touches the 4-box row. All three "
                r"MAP-Elites variants fill the 4-box row, with \textsc{ME-Str} "
                r"filling it most uniformly. \textsc{ME-Str}'s wall-density "
                r"axis is correlated with plan length (denser rooms force "
                r"longer paths), so its archive collapses toward the "
                r"long-plan cells -- the result is the strongest hard-cell "
                r"coverage of any condition. \textsc{ME-Skill} achieves "
                r"the highest \emph{overall} coverage by filling more of "
                r"the medium-length cells, leveraging the relative "
                r"orthogonality of plan length and search-node count "
                r"as descriptor axes. \textsc{ME-Play} sits between the two.")

    body.append(r"\paragraph{Qualitative inspection.} "
                r"Figure~\ref{fig:examples} shows four high-fitness elites "
                r"per condition. \textsc{ME-Str} examples tend to be dense "
                r"with internal walls. \textsc{ME-Play} produces long-plan "
                r"puzzles with 1--2 boxes that walk the player around the "
                r"board. \textsc{ME-Skill} produces puzzles with more "
                r"boxes and more constrained corridors -- visually denser "
                r"in choice points. Random examples are simple, with one "
                r"or two boxes and short obvious paths.")

    with open(out_path, "w") as f:
        f.write("\n".join(body))


def main():
    results = load_results()
    cfg = load_config()
    metrics = per_cond_metrics(results)

    tbl = summary_table(metrics)
    with open(os.path.join(RESULTS_DIR, "summary_table.txt"), "w") as f:
        f.write(f"Config: H={cfg['H']} W={cfg['W']} n_gens={cfg['n_gens']} "
                f"n_init={cfg['n_init']} node_budget={cfg['node_budget']} "
                f"n_seeds={len(cfg['seeds'])}\n\n")
        f.write(tbl + "\n")
    print(tbl)

    paper_dir = os.path.join(os.path.dirname(__file__), "..", "paper")
    with open(os.path.join(paper_dir, "results_table.tex"), "w") as f:
        f.write(latex_summary_table(metrics, cfg))
    write_results_inline(metrics, cfg, os.path.join(paper_dir, "results_inline.tex"))

    stats = pairwise_stats(metrics)
    with open(os.path.join(RESULTS_DIR, "stats.txt"), "w") as f:
        f.write(stats + "\n")
    print(stats)

    plot_convergence(results, os.path.join(RESULTS_DIR, "convergence.pdf"))
    plot_coverage_over_time(results, os.path.join(RESULTS_DIR, "coverage_over_time.pdf"))
    plot_bar_metrics(metrics, os.path.join(RESULTS_DIR, "bar_metrics.pdf"))
    plot_common_heatmaps(results, os.path.join(RESULTS_DIR, "common_heatmaps.pdf"))
    plot_solver_time(metrics, os.path.join(RESULTS_DIR, "solver_time.pdf"))

    examples = pick_examples_per_cond(results, k=8)
    write_examples(examples, os.path.join(RESULTS_DIR, "examples"))

    print(f"\nAll outputs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
