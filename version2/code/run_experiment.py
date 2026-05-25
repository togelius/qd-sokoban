"""Generic experiment driver. Runs RAND + 4 ME conditions × N seeds for
a given domain, in parallel.

Usage:
    python3 run_experiment.py --domain sokoban   [--quick] [--processes 8]
    python3 run_experiment.py --domain platformer [--quick] [--processes 8]

The output goes under `../results/<domain>/`:
    raw_runs.pkl        — list of per-run dicts (archives, eval records, log)
    summary.json        — per-run scalar summaries
    config.json         — the parameters used
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool
from typing import Any, Dict, List

import numpy as np

from qd_core import run_map_elites, run_random_search


CONDITIONS = ("RAND", "STR", "PLAY", "SKILL", "COMMON")


def _archive_to_dict(archive, render_genome) -> Dict[str, Any]:
    out = {"shape": list(archive.shape), "name": archive.name, "cells": []}
    for idx, cell in archive.grid.items():
        ev = cell.eval_result
        out["cells"].append({
            "idx": list(idx),
            "fitness": float(cell.fitness),
            "render": render_genome(cell.genome),
            "solvable": bool(getattr(ev, "solvable", False)),
            "plan_len": int(getattr(ev, "plan_len", 0) or 0),
            "nodes_expanded": int(getattr(ev, "nodes_expanded", 0) or 0),
            # domain-specific extras (any missing key just gets None)
            "n_boxes": getattr(ev, "n_boxes", None),
            "wall_density": getattr(ev, "wall_density", None),
            "n_pushes": getattr(ev, "n_pushes", None),
            "n_pits": getattr(ev, "n_pits", None),
            "n_spikes": getattr(ev, "n_spikes", None),
            "n_jumps": getattr(ev, "n_jumps", None),
            "height_std": getattr(ev, "height_std", None),
        })
    return out


def _log_to_dict(log) -> Dict[str, Any]:
    return dict(
        eval_count=log.eval_count,
        coverage=log.coverage,
        qd_score=log.qd_score,
        max_fitness=log.max_fitness,
        mean_fitness=log.mean_fitness,
        solver_calls=log.solver_calls,
        wall_time_s=log.wall_time_s,
        common_coverage=log.common_coverage,
        common_qd_score=log.common_qd_score,
        common_max_fitness=log.common_max_fitness,
        common_mean_fitness=log.common_mean_fitness,
    )


# Top-level worker (picklable for multiprocessing).
def _run_one(args):
    domain_name, cond, seed, params = args

    if domain_name == "sokoban":
        from sokoban_domain import SokobanDomain, DESCRIPTORS
        dom = SokobanDomain(H=params["H"], W=params["W"],
                            node_budget=params["node_budget"])
    elif domain_name == "platformer":
        from platformer_domain import PlatformerDomain, DESCRIPTORS
        dom = PlatformerDomain(H=params["H"], W=params["W"],
                               node_budget=params["node_budget"])
    else:
        raise ValueError(f"unknown domain {domain_name}")

    common = DESCRIPTORS["COMMON"]
    eval_records: List[dict] = []

    t0 = time.time()
    if cond == "RAND":
        archive, log = run_random_search(
            dom, common,
            n_evals=params["n_evals"],
            seed=seed,
            log_every=params["log_every"],
            eval_records=eval_records,
        )
        own_archive = archive
        common_archive = archive
    else:
        own_archive, common_archive, log = run_map_elites(
            dom,
            DESCRIPTORS[cond], common,
            n_evals=params["n_evals"],
            n_init=params["n_init"],
            seed=seed,
            log_every=params["log_every"],
            eval_records=eval_records,
        )

    wall_time = time.time() - t0

    return {
        "domain": domain_name,
        "cond": cond,
        "seed": seed,
        "params": dict(params),
        "wall_time_s": wall_time,
        "own_archive": _archive_to_dict(own_archive, dom.render_genome),
        "common_archive": _archive_to_dict(common_archive, dom.render_genome),
        "log": _log_to_dict(log),
        "eval_records": eval_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["sokoban", "platformer"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.domain == "sokoban":
        if args.quick:
            params = dict(H=7, W=7, n_evals=1500, n_init=100,
                          node_budget=3000, log_every=150)
            seeds = list(range(3))
        else:
            params = dict(H=8, W=8, n_evals=5500, n_init=250,
                          node_budget=8000, log_every=250)
            seeds = list(range(20))
    else:  # platformer
        if args.quick:
            params = dict(H=10, W=20, n_evals=1500, n_init=100,
                          node_budget=3000, log_every=150)
            seeds = list(range(3))
        else:
            params = dict(H=10, W=24, n_evals=5000, n_init=250,
                          node_budget=4000, log_every=250)
            seeds = list(range(20))

    outdir = args.outdir or f"../results/{args.domain}"
    os.makedirs(outdir, exist_ok=True)
    config = dict(domain=args.domain, conditions=list(CONDITIONS),
                  seeds=seeds, params=params, quick=args.quick)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    jobs = [(args.domain, cond, seed, params)
            for cond in CONDITIONS for seed in seeds]

    print(f"[{args.domain}] launching {len(jobs)} runs over {args.processes} processes")
    print(f"  params: {params}")
    t0 = time.time()
    results = []
    with Pool(processes=args.processes) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs)):
            results.append(r)
            cov_common = len(r["common_archive"]["cells"])
            qd_common = sum(c["fitness"] for c in r["common_archive"]["cells"])
            print(f"  [{i+1:>3}/{len(jobs)}] {r['cond']:<6} seed={r['seed']:<2} "
                  f"cc={cov_common:<3} cqd={qd_common:.2f} "
                  f"t={r['wall_time_s']:.1f}s")
    elapsed = time.time() - t0
    print(f"[{args.domain}] done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    with open(os.path.join(outdir, "raw_runs.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"  wrote raw_runs.pkl ({os.path.getsize(os.path.join(outdir, 'raw_runs.pkl'))/1024/1024:.1f} MB)")

    summary = []
    for r in results:
        common = r["common_archive"]["cells"]
        own = r["own_archive"]["cells"]
        cov = len(common)
        qd = sum(c["fitness"] for c in common)
        max_fit = max((c["fitness"] for c in common), default=0.0)
        mean_fit = qd / cov if cov else 0.0
        own_cov = len(own)
        own_qd = sum(c["fitness"] for c in own)
        n_solver = r["log"]["solver_calls"][-1]
        summary.append(dict(
            domain=r["domain"], cond=r["cond"], seed=r["seed"],
            common_coverage=cov, common_qd_score=qd,
            common_max_fitness=max_fit, common_mean_fitness=mean_fit,
            own_coverage=own_cov, own_qd_score=own_qd,
            n_solver_calls=n_solver,
            wall_time_s=r["wall_time_s"],
        ))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote summary.json ({len(summary)} runs)")


if __name__ == "__main__":
    main()
