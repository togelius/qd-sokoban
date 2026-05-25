"""Main experiment driver.

Runs 4 conditions × N seeds and saves results to ../results/. Uses
multiprocessing for parallelism over runs.

Usage:
    python3 experiment.py [--quick]

With --quick: fewer gens + fewer seeds for sanity checking.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

from descriptors import (
    STRUCT_DESC, PLAY_DESC, SKILL_DESC, COMMON_DESC,
    common_cell, COMMON_SHAPE,
)
from qd import (
    run_map_elites, run_random_search,
    archive_to_dict, log_to_dict,
)


# Map condition name -> (descriptor dict or None for random)
CONDITIONS = {
    "STR":   STRUCT_DESC,
    "PLAY":  PLAY_DESC,
    "SKILL": SKILL_DESC,
    "RAND":  None,
}


def _run_one(args):
    """Top-level for pickleability under multiprocessing."""
    cond, seed, H, W, n_gens, n_init, node_budget, log_every = args
    desc = CONDITIONS[cond]
    if desc is None:
        archive, log = run_random_search(
            n_generations=n_gens, n_init=n_init,
            H=H, W=W, seed=seed,
            node_budget=node_budget, log_every=log_every,
        )
        common = archive  # for RAND, own archive == common archive
    else:
        archive, log, common = run_map_elites(
            descriptor_fn=desc["to_cell"],
            archive_shape=desc["shape"],
            archive_name=cond,
            n_generations=n_gens, n_init=n_init,
            H=H, W=W, seed=seed,
            node_budget=node_budget, log_every=log_every,
        )
    return {
        "cond": cond,
        "seed": seed,
        "params": dict(H=H, W=W, n_gens=n_gens, n_init=n_init,
                       node_budget=node_budget, log_every=log_every),
        "archive": archive_to_dict(archive),
        "common": archive_to_dict(common),
        "log": log_to_dict(log),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()

    if args.quick:
        seeds = list(range(3))
        n_gens = 1000
        n_init = 100
        node_budget = 3000
        H, W = 7, 7
    else:
        seeds = list(range(10))
        n_gens = 5000
        n_init = 200
        node_budget = 8000
        H, W = 8, 8

    log_every = max(1, n_gens // 25)

    os.makedirs(args.outdir, exist_ok=True)
    config = {
        "H": H, "W": W, "n_gens": n_gens, "n_init": n_init,
        "node_budget": node_budget, "log_every": log_every,
        "seeds": seeds, "conditions": list(CONDITIONS.keys()),
        "quick": args.quick,
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    jobs = []
    for cond in CONDITIONS:
        for seed in seeds:
            jobs.append((cond, seed, H, W, n_gens, n_init, node_budget, log_every))

    print(f"Launching {len(jobs)} runs over {args.processes} processes")
    t0 = time.time()
    results = []
    with Pool(processes=args.processes) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs)):
            results.append(r)
            cov = r["common"]["cells"]
            qd = sum(c["fit"] for c in cov)
            print(f"  [{i+1:>2}/{len(jobs)}] cond={r['cond']} seed={r['seed']} "
                  f"common_cov={len(cov)} common_qd={qd:.1f} "
                  f"wall_t={r['log']['wall_time_s'][-1]:.1f}s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    out_path = os.path.join(args.outdir, "raw_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} results to {out_path}")

    # Also save a small summary JSON
    summary = []
    for r in results:
        common_qd = sum(c["fit"] for c in r["common"]["cells"])
        summary.append(dict(
            cond=r["cond"], seed=r["seed"],
            common_coverage=len(r["common"]["cells"]),
            common_qd_score=common_qd,
            common_max_fit=max((c["fit"] for c in r["common"]["cells"]), default=0.0),
            wall_time_s=r["log"]["wall_time_s"][-1],
            n_solver_calls=r["log"]["n_solver_calls"][-1],
        ))
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary.json")


if __name__ == "__main__":
    main()
