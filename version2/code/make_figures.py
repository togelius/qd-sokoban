"""Generate the publication figures.

Outputs (per domain, into ../results/<domain>/figures/):
    convergence.pdf         coverage / qd vs evaluations, all conditions
    bars.pdf                final coverage / qd bars, per condition
    heatmaps_common.pdf     per-condition mean common-cell frequency
    mechanism.pdf           per-condition mean log_nodes of retained elites
    wall_time.pdf           wall-clock vs condition

Plus a combined cross-domain figure:
    ../results/cross_domain_summary.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["RAND", "STR", "PLAY", "SKILL", "COMMON"]
COND_COLORS = {
    "RAND":   "#888888",
    "STR":    "#d62728",
    "PLAY":   "#1f77b4",
    "SKILL":  "#2ca02c",
    "COMMON": "#9467bd",
}


def load_runs(outdir):
    with open(os.path.join(outdir, "raw_runs.pkl"), "rb") as f:
        return pickle.load(f)


def fig_convergence(runs, outpath, domain_label=""):
    """common_coverage and common_qd_score vs eval_count, mean ± std band."""
    by_cond = defaultdict(list)
    eval_axes = defaultdict(list)
    for r in runs:
        cond = r["cond"]
        e = r["log"]["eval_count"]
        cov = r["log"]["common_coverage"]
        qd = r["log"]["common_qd_score"]
        by_cond[cond].append((np.asarray(e), np.asarray(cov), np.asarray(qd)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric_idx, metric_label in [
        (axes[0], 1, "common-archive coverage"),
        (axes[1], 2, "common-archive QD-score"),
    ]:
        for cond in CONDITIONS:
            if cond not in by_cond:
                continue
            series = by_cond[cond]
            # all runs have same eval_count axis
            e = series[0][0]
            arr = np.stack([s[metric_idx] for s in series])
            mean = arr.mean(0)
            sd = arr.std(0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(e, mean, color=COND_COLORS[cond], label=cond, lw=1.6)
            ax.fill_between(e, mean - sd, mean + sd,
                            color=COND_COLORS[cond], alpha=0.18, linewidth=0)
        ax.set_xlabel("evaluations")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=5, fontsize=8, loc="lower right", frameon=False)
    if domain_label:
        fig.suptitle(domain_label, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def fig_bars(runs, outpath, domain_label=""):
    """Final common-coverage and common-qd bars with seed-level dots."""
    by_cond_cov = defaultdict(list)
    by_cond_qd = defaultdict(list)
    by_cond_t = defaultdict(list)
    for r in runs:
        cells = r["common_archive"]["cells"]
        cond = r["cond"]
        by_cond_cov[cond].append(len(cells))
        by_cond_qd[cond].append(sum(c["fitness"] for c in cells))
        by_cond_t[cond].append(r["wall_time_s"])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, dat, label in [
        (axes[0], by_cond_cov, "common coverage (cells)"),
        (axes[1], by_cond_qd,  "common QD-score"),
        (axes[2], by_cond_t,   "wall-clock time (s)"),
    ]:
        xs = []
        for x_idx, cond in enumerate(CONDITIONS):
            if cond not in dat:
                continue
            vals = np.asarray(dat[cond], dtype=float)
            mean = vals.mean()
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.bar(x_idx, mean, color=COND_COLORS[cond], alpha=0.6, width=0.6)
            ax.errorbar(x_idx, mean, yerr=sd, color="k", capsize=3, lw=0.8)
            # Jittered dots
            jitter = np.random.RandomState(0).normal(0, 0.06, len(vals))
            ax.scatter(np.full(len(vals), x_idx) + jitter, vals,
                       color="k", alpha=0.55, s=8, zorder=3)
            xs.append((x_idx, cond))
        ax.set_xticks([x for x, _ in xs])
        ax.set_xticklabels([c for _, c in xs])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, axis="y")
    if domain_label:
        fig.suptitle(domain_label, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def fig_heatmaps(runs, outpath, domain_label=""):
    """Per-condition heatmap of mean common-cell fill across seeds."""
    if not runs:
        return
    shape = tuple(runs[0]["common_archive"]["shape"])
    by_cond = defaultdict(lambda: (np.zeros(shape, dtype=float), 0))
    for r in runs:
        arr, n = by_cond[r["cond"]]
        arr = arr.copy()
        for c in r["common_archive"]["cells"]:
            i, j = c["idx"]
            arr[i, j] += 1.0
        by_cond[r["cond"]] = (arr, n + 1)

    # Axis labels for the COMMON projection (domain-specific).
    if domain_label.lower().startswith("sokoban"):
        y_label = "# boxes"
        x_label = "plan-length bin"
        y_ticks = ["1", "2", "3", "4"]
    else:
        y_label = "# pits"
        x_label = "plan-length bin"
        y_ticks = ["0", "1", "2", "3", "4", "5", "6-7", "8+"]

    conds = [c for c in CONDITIONS if c in by_cond]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.4 * len(conds), 2.6),
                             sharey=True)
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        arr, n = by_cond[cond]
        freq = arr / max(n, 1)
        im = ax.imshow(freq, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="upper")
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel(x_label)
        if cond == conds[0]:
            ax.set_ylabel(y_label)
            ax.set_yticks(range(shape[0]))
            ax.set_yticklabels(y_ticks)
    fig.colorbar(im, ax=axes, fraction=0.02, label="fill freq across seeds")
    if domain_label:
        fig.suptitle(f"{domain_label}: per-condition common-cell fill frequency",
                     y=1.05, fontsize=10)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def fig_mechanism(runs, outpath, domain_label=""):
    """log10(nodes) distribution of the retained elites, per condition.
    The 'retained' set = the final common-archive cells. We compare these
    to the distribution of all *evaluated* solvable individuals (the
    'evaluated' pool)."""
    eval_log = defaultdict(list)
    retain_log = defaultdict(list)
    for r in runs:
        cond = r["cond"]
        for e in r["eval_records"]:
            if e["solvable"] and e["nodes_expanded"] > 0:
                eval_log[cond].append(np.log10(e["nodes_expanded"]))
        for c in r["common_archive"]["cells"]:
            if c["solvable"] and c["nodes_expanded"] > 0:
                retain_log[cond].append(np.log10(c["nodes_expanded"]))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x_positions = np.arange(len([c for c in CONDITIONS if c in retain_log]))
    width = 0.36
    labels = []
    for i, cond in enumerate(c for c in CONDITIONS if c in retain_log):
        ev = np.asarray(eval_log[cond])
        rt = np.asarray(retain_log[cond])
        if len(ev) == 0 or len(rt) == 0:
            continue
        ev_med, ev_lo, ev_hi = np.median(ev), np.percentile(ev, 25), np.percentile(ev, 75)
        rt_med, rt_lo, rt_hi = np.median(rt), np.percentile(rt, 25), np.percentile(rt, 75)
        x = x_positions[i]
        # boxlets: evaluated (left, gray) vs retained (right, colored)
        ax.bar(x - width/2, ev_med, width, color="#cccccc",
               edgecolor="k", label=("evaluated" if i == 0 else None))
        ax.errorbar(x - width/2, ev_med, yerr=[[ev_med - ev_lo], [ev_hi - ev_med]],
                    color="k", capsize=3, lw=0.8)
        ax.bar(x + width/2, rt_med, width, color=COND_COLORS[cond],
               alpha=0.8, edgecolor="k",
               label=("retained" if i == 0 else None))
        ax.errorbar(x + width/2, rt_med, yerr=[[rt_med - rt_lo], [rt_hi - rt_med]],
                    color="k", capsize=3, lw=0.8)
        labels.append(cond)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log10(A* nodes expanded)")
    ax.set_title("Implicit hardness selection: evaluated vs retained elites"
                 + (f" — {domain_label}" if domain_label else ""))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run_make_figures(outdir, domain_label):
    runs = load_runs(outdir)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_convergence(runs, os.path.join(figdir, "convergence.pdf"), domain_label)
    fig_bars(runs, os.path.join(figdir, "bars.pdf"), domain_label)
    fig_heatmaps(runs, os.path.join(figdir, "heatmaps_common.pdf"), domain_label)
    fig_mechanism(runs, os.path.join(figdir, "mechanism.pdf"), domain_label)
    print(f"  wrote figures to {figdir}")


def fig_cross_domain(soko_runs, plat_runs, outpath):
    """Bar chart of common_coverage per condition for both domains side
    by side. The cross-domain headline figure."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.35
    x = np.arange(len(CONDITIONS))
    soko_means = []
    soko_sds = []
    plat_means = []
    plat_sds = []
    for cond in CONDITIONS:
        s = [len(r["common_archive"]["cells"]) for r in soko_runs if r["cond"] == cond]
        p = [len(r["common_archive"]["cells"]) for r in plat_runs if r["cond"] == cond]
        soko_means.append(np.mean(s) if s else 0)
        soko_sds.append(np.std(s, ddof=1) if len(s) > 1 else 0)
        plat_means.append(np.mean(p) if p else 0)
        plat_sds.append(np.std(p, ddof=1) if len(p) > 1 else 0)
    soko_max = 40
    plat_max = 80
    ax.bar(x - width / 2, np.asarray(soko_means) / soko_max, width,
           yerr=np.asarray(soko_sds) / soko_max,
           color="#1f77b4", alpha=0.8, label="Sokoban (frac of 40)",
           capsize=3)
    ax.bar(x + width / 2, np.asarray(plat_means) / plat_max, width,
           yerr=np.asarray(plat_sds) / plat_max,
           color="#ff7f0e", alpha=0.8, label="Platformer (frac of 80)",
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("common-archive coverage (normalised)")
    ax.set_title("Cross-domain coverage by condition")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["sokoban", "platformer"])
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.combined:
        soko = load_runs("../results/sokoban")
        plat = load_runs("../results/platformer")
        os.makedirs("../results", exist_ok=True)
        fig_cross_domain(soko, plat, "../results/cross_domain_summary.pdf")
        print("wrote ../results/cross_domain_summary.pdf")
    else:
        run_make_figures(args.outdir or f"../results/{args.domain}",
                         args.domain.capitalize())


if __name__ == "__main__":
    main()
