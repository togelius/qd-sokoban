"""For each condition, compute:
  - fraction of evaluations that are infeasible (well_formed but unsolved)
  - mean solver work *per evaluation*, including timed-out failures
  - mean solver work *over solvable only*

The cross-condition gap on the first metric is what should explain
STR's wall-clock overhead in Sokoban.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def compute(outdir: str) -> dict:
    with open(os.path.join(outdir, "raw_runs.pkl"), "rb") as f:
        runs = pickle.load(f)
    by_cond = defaultdict(lambda: {"n_eval": 0, "n_well_formed": 0,
                                    "n_solvable": 0, "n_unsolvable_attempted": 0,
                                    "logn_all_attempted": [],  # incl. timeouts
                                    "logn_solvable_only": []})
    for r in runs:
        cond = r["cond"]
        for e in r["eval_records"]:
            d = by_cond[cond]
            d["n_eval"] += 1
            if e["well_formed"]:
                d["n_well_formed"] += 1
            if e["solvable"]:
                d["n_solvable"] += 1
                if e["nodes_expanded"] > 0:
                    d["logn_solvable_only"].append(np.log10(e["nodes_expanded"]))
            elif e["well_formed"] and e["nodes_expanded"] > 0:
                # well-formed but unsolved — solver ran but timed out or
                # proved infeasibility
                d["n_unsolvable_attempted"] += 1
            if e["nodes_expanded"] > 0:
                d["logn_all_attempted"].append(np.log10(e["nodes_expanded"]))

    out = {}
    for cond, d in by_cond.items():
        n_eval = d["n_eval"]
        out[cond] = dict(
            n_eval=n_eval,
            frac_well_formed=d["n_well_formed"] / n_eval,
            frac_solvable=d["n_solvable"] / n_eval,
            frac_unsolvable_attempted=d["n_unsolvable_attempted"] / n_eval,
            mean_logn_all_attempted=float(np.mean(d["logn_all_attempted"])) if d["logn_all_attempted"] else None,
            mean_logn_solvable=float(np.mean(d["logn_solvable_only"])) if d["logn_solvable_only"] else None,
            # Mean over all_attempted divides by attempts including timeouts;
            # the gap with solvable-only is the failure-driven overhead.
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["sokoban", "platformer"])
    args = ap.parse_args()
    outdir = f"../results/{args.domain}"
    res = compute(outdir)
    print(f"=== {args.domain} ===")
    print(f"{'cond':<8} {'n':>8} {'%wf':>6} {'%solv':>6} {'%unsolv':>7} "
          f"{'logN_all':>9} {'logN_solv':>10}")
    for cond, d in res.items():
        print(f"{cond:<8} {d['n_eval']:>8} "
              f"{100*d['frac_well_formed']:>5.1f}% "
              f"{100*d['frac_solvable']:>5.1f}% "
              f"{100*d['frac_unsolvable_attempted']:>6.1f}% "
              f"{d['mean_logn_all_attempted']:>9.3f} "
              f"{d['mean_logn_solvable']:>10.3f}")
    with open(os.path.join(outdir, "failure_breakdown.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
