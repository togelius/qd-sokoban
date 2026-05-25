"""Auto-generate LaTeX tables and inline-result macros for the paper.

Outputs (into ../paper/):
    results_table.tex     : tabular Sokoban + Platformer summary
    results_pairwise.tex  : Mann-Whitney pairwise table per metric (compact)
    results_inline.tex    : \\newcommand wrappers for in-text numbers
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from analyze import metric_table, load_runs, pairwise_table
from stats_utils import bootstrap_ci


COND_ORDER = ["RAND", "STR", "PLAY", "SKILL", "COMMON"]


def _fmt(x, prec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{prec}f}"


def make_summary_table(domains: Dict[str, dict]) -> str:
    """Two-block tabular: one block per domain, three metrics
    (common_coverage, common_qd_score, wall_time_s)."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-condition summary statistics. mean$\pm$SD across 20 seeds; bootstrap 95\% CIs for the mean in parentheses.}")
    lines.append(r"\label{tab:summary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l rrr rrr}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{Common coverage} & \multicolumn{3}{c}{Common QD-score} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r"domain & cond & mean$\pm$SD & 95\% CI & wall (s) & mean$\pm$SD & 95\% CI & solver calls \\")
    lines.append(r"\midrule")
    for dom_name, table in domains.items():
        for cond in COND_ORDER:
            if cond not in table["common_coverage"]:
                continue
            cov = np.asarray(table["common_coverage"][cond], dtype=float)
            qd = np.asarray(table["common_qd_score"][cond], dtype=float)
            wt = np.asarray(table["wall_time_s"][cond], dtype=float)
            sc = np.asarray(table["n_solver_calls"][cond], dtype=float)
            cov_pt, cov_lo, cov_hi = bootstrap_ci(cov)
            qd_pt, qd_lo, qd_hi = bootstrap_ci(qd)
            lines.append(
                f"{dom_name} & \\textsf{{{cond}}} & "
                f"{cov.mean():.2f}$\\pm${cov.std(ddof=1):.2f} & "
                f"[{cov_lo:.2f}, {cov_hi:.2f}] & "
                f"{wt.mean():.1f} & "
                f"{qd.mean():.2f}$\\pm${qd.std(ddof=1):.2f} & "
                f"[{qd_lo:.2f}, {qd_hi:.2f}] & "
                f"{sc.mean():.0f}"
                f" \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_pairwise_table(domains: Dict[str, dict], metric: str = "common_coverage",
                        caption: str = None) -> str:
    """Compact pairwise table for a single metric, both domains side-by-side."""
    caption = caption or f"Pairwise Mann-Whitney $U$ tests (Holm-corrected) for {metric}. Cells show $p_{{\\text{{adj}}}}$ / Cliff's $\\delta$. \\textbf{{Bold}} if $p_{{\\text{{adj}}}} < 0.05$."
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:pairwise-" + metric + r"}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")

    for dom_name, table in domains.items():
        if metric not in table:
            continue
        groups = table[metric]
        if not groups:
            continue
        rows = pairwise_table(groups, metric_label=metric)
        # Find present conditions in canonical order
        present = [c for c in COND_ORDER if c in groups]
        lines.append(r"\noindent\textbf{" + dom_name + r"}\\")
        lines.append(r"\begin{tabular}{l " + " ".join(["c"] * len(present)) + r"}")
        lines.append(r"\toprule")
        lines.append(" & " + " & ".join(present) + r" \\")
        lines.append(r"\midrule")
        # build a lookup
        lookup = {(r.a, r.b): r for r in rows}
        for i, a in enumerate(present):
            cells = [a]
            for j, b in enumerate(present):
                if a == b:
                    cells.append("--")
                else:
                    key = (a, b) if (a, b) in lookup else (b, a)
                    if key in lookup:
                        r = lookup[key]
                        # If we accessed via (b, a), flip the sign for clarity
                        d = r.cliffs_delta if key == (a, b) else -r.cliffs_delta
                        txt = f"{r.p_adj:.3f}/{d:+.2f}"
                        if r.p_adj < 0.05:
                            txt = r"\textbf{" + txt + "}"
                        cells.append(txt)
                    else:
                        cells.append("--")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\vspace{4pt}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_inline_macros(domains: Dict[str, dict]) -> str:
    """\\newcommand wrappers so paper.tex can refer to fresh numbers via macros."""
    out = []
    out.append(r"% Auto-generated from analyze.py — do not edit by hand.")
    for dom_name, table in domains.items():
        prefix = "\\result" + dom_name.capitalize()
        for cond in COND_ORDER:
            if cond not in table["common_coverage"]:
                continue
            cov = np.asarray(table["common_coverage"][cond], dtype=float)
            qd = np.asarray(table["common_qd_score"][cond], dtype=float)
            wt = np.asarray(table["wall_time_s"][cond], dtype=float)
            out.append(f"\\newcommand{{{prefix}{cond}Cov}}{{{cov.mean():.1f}$\\pm${cov.std(ddof=1):.1f}}}")
            out.append(f"\\newcommand{{{prefix}{cond}QD}}{{{qd.mean():.2f}$\\pm${qd.std(ddof=1):.2f}}}")
            out.append(f"\\newcommand{{{prefix}{cond}WT}}{{{wt.mean():.1f}$\\pm${wt.std(ddof=1):.1f}}}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soko_dir", default="../results/sokoban")
    ap.add_argument("--plat_dir", default="../results/platformer")
    ap.add_argument("--out_dir",  default="../paper")
    args = ap.parse_args()

    soko = load_runs(args.soko_dir)
    plat = load_runs(args.plat_dir)
    domains = {
        "Sokoban":    metric_table(soko),
        "Platformer": metric_table(plat),
    }

    tab = make_summary_table(domains)
    with open(os.path.join(args.out_dir, "results_table.tex"), "w") as f:
        f.write(tab + "\n")
    pw = make_pairwise_table(domains, metric="common_coverage")
    with open(os.path.join(args.out_dir, "results_pairwise.tex"), "w") as f:
        f.write(pw + "\n")
    inline = make_inline_macros(domains)
    with open(os.path.join(args.out_dir, "results_inline.tex"), "w") as f:
        f.write(inline)
    print("wrote results_table.tex, results_pairwise.tex, results_inline.tex")


if __name__ == "__main__":
    main()
