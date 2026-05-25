"""Tiny smoke run: 500 evals per condition, single seed, prints summary."""

from __future__ import annotations

import time

from qd_core import run_map_elites, run_random_search
from sokoban_domain import SokobanDomain, DESCRIPTORS


def main():
    dom = SokobanDomain(H=7, W=7, node_budget=3000)
    common = DESCRIPTORS["COMMON"]
    n_evals = 500

    print(f"{'cond':<8} {'cov':>4} {'qd':>6} {'maxf':>6} {'t(s)':>6}")
    for cond in ["RAND", "STR", "PLAY", "SKILL", "COMMON"]:
        t0 = time.time()
        if cond == "RAND":
            arch, log = run_random_search(dom, common, n_evals=n_evals, seed=0)
        else:
            arch, com, log = run_map_elites(
                dom, DESCRIPTORS[cond], common,
                n_evals=n_evals, n_init=50, seed=0,
            )
            arch = com  # report common-projection metrics
        print(f"{cond:<8} {arch.coverage():>4} {arch.qd_score():>6.2f} "
              f"{arch.max_fitness():>6.3f} {time.time()-t0:>6.1f}")


if __name__ == "__main__":
    main()
