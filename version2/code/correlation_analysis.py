"""For each domain, compute the Spearman correlation between each
descriptor's coordinates and the solver-effort log_nodes, using all
solvable evaluations from the run. This directly demonstrates the
mechanism: descriptors whose coordinates correlate with hardness
impose implicit hardness-selection pressure.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from stats_utils import spearman_with_ci


def compute(outdir: str) -> dict:
    with open(os.path.join(outdir, "raw_runs.pkl"), "rb") as f:
        runs = pickle.load(f)

    # For each condition's runs, gather (descriptor_cell_i, descriptor_cell_j,
    # log_nodes) tuples over all solvable evaluations.
    by_cond = defaultdict(lambda: {"axis1": [], "axis2": [], "logn": []})
    for r in runs:
        cond = r["cond"]
        for e in r["eval_records"]:
            if not e["solvable"] or e["nodes_expanded"] <= 0:
                continue
            cell = e["descriptor_cell"]
            if cell is None:
                continue
            i, j = cell
            by_cond[cond]["axis1"].append(i)
            by_cond[cond]["axis2"].append(j)
            by_cond[cond]["logn"].append(np.log10(e["nodes_expanded"]))

    out = {}
    for cond, data in by_cond.items():
        if len(data["axis1"]) < 10:
            continue
        rho1, lo1, hi1, p1 = spearman_with_ci(data["axis1"], data["logn"])
        rho2, lo2, hi2, p2 = spearman_with_ci(data["axis2"], data["logn"])
        out[cond] = dict(
            n=len(data["axis1"]),
            spearman_axis1=dict(rho=rho1, ci=[lo1, hi1], p=p1),
            spearman_axis2=dict(rho=rho2, ci=[lo2, hi2], p=p2),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["sokoban", "platformer"])
    args = ap.parse_args()
    outdir = f"../results/{args.domain}"
    res = compute(outdir)
    print(f"=== {args.domain} ===")
    print(f"{'cond':<8} {'n':>7} {'rho_ax1':>9} {'95% CI':>16} {'p':>9} | "
          f"{'rho_ax2':>9} {'95% CI':>16} {'p':>9}")
    for cond, d in res.items():
        r1 = d["spearman_axis1"]; r2 = d["spearman_axis2"]
        print(f"{cond:<8} {d['n']:>7} "
              f"{r1['rho']:>+9.3f} [{r1['ci'][0]:+5.2f},{r1['ci'][1]:+5.2f}] {r1['p']:>9.2g} | "
              f"{r2['rho']:>+9.3f} [{r2['ci'][0]:+5.2f},{r2['ci'][1]:+5.2f}] {r2['p']:>9.2g}")
    with open(os.path.join(outdir, "descriptor_hardness_correlation.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
