"""Per-domain analysis: load raw runs, compute summary tables, Mann-Whitney
+ Holm-Bonferroni pairwise tests, Cliff's δ effect sizes, mechanism analyses.

Usage:
    python3 analyze.py --domain sokoban
    python3 analyze.py --domain platformer
    python3 analyze.py --combined    # cross-domain table

Outputs to ../results/<domain>/:
    stats_summary.txt         — human-readable per-metric table
    stats_pairwise.json       — full pairwise comparison data
    stats_pairwise.txt        — human-readable pairwise table
    mechanism.json            — descriptor-vs-hardness correlations
    common_frequency.npy      — per-cell fill frequency across seeds
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from stats_utils import (PairwiseStat, bootstrap_ci, cliffs_delta,
                         pairwise_table, spearman_with_ci)


METRICS = [
    "common_coverage",
    "common_qd_score",
    "common_mean_fitness",
    "common_max_fitness",
    "wall_time_s",
    "n_solver_calls",
]


def load_runs(outdir: str):
    with open(os.path.join(outdir, "raw_runs.pkl"), "rb") as f:
        return pickle.load(f)


def metric_table(runs) -> Dict[str, Dict[str, List[float]]]:
    """Return dict of metric_name -> {cond_name -> [values across seeds]}."""
    out: Dict[str, Dict[str, List[float]]] = {m: defaultdict(list) for m in METRICS}
    for r in runs:
        cond = r["cond"]
        common = r["common_archive"]["cells"]
        cov = len(common)
        qd = sum(c["fitness"] for c in common)
        mean_fit = qd / cov if cov else 0.0
        max_fit = max((c["fitness"] for c in common), default=0.0)
        wt = r["wall_time_s"]
        nsc = r["log"]["solver_calls"][-1]
        out["common_coverage"][cond].append(cov)
        out["common_qd_score"][cond].append(qd)
        out["common_mean_fitness"][cond].append(mean_fit)
        out["common_max_fitness"][cond].append(max_fit)
        out["wall_time_s"][cond].append(wt)
        out["n_solver_calls"][cond].append(nsc)
    return {m: dict(v) for m, v in out.items()}


def summary_text(table) -> str:
    lines = []
    lines.append(f"{'metric':<22} {'cond':<8} {'n':<3} {'mean':>8} {'std':>7} {'med':>7} {'ci95':>20}")
    lines.append("-" * 80)
    for m in METRICS:
        for cond in ["RAND", "STR", "PLAY", "SKILL", "COMMON"]:
            xs = table[m].get(cond, [])
            if not xs:
                continue
            xs = np.asarray(xs, dtype=float)
            pt, lo, hi = bootstrap_ci(xs)
            lines.append(f"{m:<22} {cond:<8} {len(xs):<3} {xs.mean():>8.3f} "
                         f"{xs.std(ddof=1):>7.3f} {np.median(xs):>7.3f} "
                         f"[{lo:>7.3f}, {hi:>7.3f}]")
        lines.append("")
    return "\n".join(lines)


def pairwise_text(table) -> str:
    lines = []
    for m in METRICS:
        if not table[m]:
            continue
        rows = pairwise_table(table[m], metric_label=m)
        lines.append(f"=== {m} ===")
        lines.append(f"  {'A':>6} vs {'B':<6}  {'meanA':>8} {'meanB':>8}  "
                     f"{'p':>9} {'p_adj':>9}  {'cliffsd':>7}  {'effect':<11}")
        for r in rows:
            sig = " *" if r.p_adj < 0.05 else ""
            lines.append(f"  {r.a:>6} vs {r.b:<6}  "
                         f"{r.mean_a:>8.3f} {r.mean_b:>8.3f}  "
                         f"{r.p_value:>9.3g} {r.p_adj:>9.3g}  "
                         f"{r.cliffs_delta:>+7.2f}  {r.interpretation:<11}{sig}")
        lines.append("")
    return "\n".join(lines)


def pairwise_to_dicts(table) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for m in METRICS:
        if not table[m]:
            continue
        rows = pairwise_table(table[m], metric_label=m)
        out[m] = [dict(
            a=r.a, b=r.b, mean_a=r.mean_a, mean_b=r.mean_b,
            median_a=r.median_a, median_b=r.median_b,
            p_value=r.p_value, p_adj=r.p_adj,
            cliffs_delta=r.cliffs_delta, interpretation=r.interpretation,
            direction=r.direction,
        ) for r in rows]
    return out


def common_frequency(runs, shape) -> Dict[str, np.ndarray]:
    """For each condition, how often each common-archive cell gets filled
    across seeds (count / n_seeds)."""
    counts = defaultdict(lambda: np.zeros(shape, dtype=float))
    n_seeds = defaultdict(int)
    for r in runs:
        n_seeds[r["cond"]] += 1
        for cell in r["common_archive"]["cells"]:
            i, j = cell["idx"]
            counts[r["cond"]][i, j] += 1.0
    return {c: counts[c] / max(n_seeds[c], 1) for c in counts}


def mechanism_analysis(runs) -> dict:
    """For each condition, compute:
      - mean log_nodes of *retained* elites in the common archive
      - Spearman correlation between own-descriptor cell index and log_nodes
        (computed across the eval_records, restricted to *solvable* evaluations)
      - mean log_nodes of cell-winners minus mean log_nodes of all evaluations
        (the 'retention bias').
    """
    out = {"by_cond": {}}
    for cond in ["RAND", "STR", "PLAY", "SKILL", "COMMON"]:
        cond_runs = [r for r in runs if r["cond"] == cond]
        if not cond_runs:
            continue
        all_log_nodes_evaluated = []
        all_log_nodes_retained = []
        elite_log_nodes_per_seed = []
        for r in cond_runs:
            evals = r["eval_records"]
            for e in evals:
                if e["solvable"] and e["nodes_expanded"] > 0:
                    all_log_nodes_evaluated.append(np.log10(e["nodes_expanded"]))
            elites = r["common_archive"]["cells"]
            seed_logs = []
            for c in elites:
                if c["solvable"] and c["nodes_expanded"] > 0:
                    seed_logs.append(np.log10(c["nodes_expanded"]))
                    all_log_nodes_retained.append(np.log10(c["nodes_expanded"]))
            if seed_logs:
                elite_log_nodes_per_seed.append(float(np.mean(seed_logs)))
        out["by_cond"][cond] = {
            "n_evals_total": len(all_log_nodes_evaluated),
            "mean_logn_evaluated": float(np.mean(all_log_nodes_evaluated))
                                   if all_log_nodes_evaluated else None,
            "mean_logn_retained": float(np.mean(all_log_nodes_retained))
                                   if all_log_nodes_retained else None,
            "retention_bias": (float(np.mean(all_log_nodes_retained) -
                                     np.mean(all_log_nodes_evaluated))
                               if all_log_nodes_evaluated and all_log_nodes_retained
                               else None),
            "elite_logn_per_seed": elite_log_nodes_per_seed,
        }
    # Pairwise test of "elite_logn_per_seed" across conditions.
    groups = {c: out["by_cond"][c]["elite_logn_per_seed"]
              for c in out["by_cond"]
              if out["by_cond"][c]["elite_logn_per_seed"]}
    if len(groups) >= 2:
        rows = pairwise_table(groups, metric_label="elite_log_nodes")
        out["elite_logn_pairwise"] = [dict(
            a=r.a, b=r.b, mean_a=r.mean_a, mean_b=r.mean_b,
            p_value=r.p_value, p_adj=r.p_adj,
            cliffs_delta=r.cliffs_delta,
            interpretation=r.interpretation, direction=r.direction,
        ) for r in rows]
    return out


def run_analyze(outdir: str):
    runs = load_runs(outdir)
    print(f"[{outdir}] loaded {len(runs)} runs")
    table = metric_table(runs)
    txt = summary_text(table)
    with open(os.path.join(outdir, "stats_summary.txt"), "w") as f:
        f.write(txt)
    print(txt)
    pw_text = pairwise_text(table)
    with open(os.path.join(outdir, "stats_pairwise.txt"), "w") as f:
        f.write(pw_text)
    pw_data = pairwise_to_dicts(table)
    with open(os.path.join(outdir, "stats_pairwise.json"), "w") as f:
        json.dump(pw_data, f, indent=2)
    # frequency arrays — use first run to infer shape
    if runs:
        shape = tuple(runs[0]["common_archive"]["shape"])
        freq = common_frequency(runs, shape)
        np.savez(os.path.join(outdir, "common_frequency.npz"),
                 **{c: freq[c] for c in freq})
    # mechanism
    mech = mechanism_analysis(runs)
    with open(os.path.join(outdir, "mechanism.json"), "w") as f:
        json.dump(mech, f, indent=2)
    print(f"  wrote stats_summary.txt, stats_pairwise.txt/json, "
          f"common_frequency.npz, mechanism.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["sokoban", "platformer"])
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.combined:
        # combine summaries from both domains
        for d in ["sokoban", "platformer"]:
            run_analyze(args.outdir or f"../results/{d}")
    else:
        run_analyze(args.outdir or f"../results/{args.domain}")


if __name__ == "__main__":
    main()
