"""Smoke run for platformer: tests game, solver, generator, and QD loop."""

from __future__ import annotations

import random
import time

from qd_core import run_map_elites, run_random_search
from platformer_domain import PlatformerDomain, DESCRIPTORS
from platformer_solver import solve


def test_solver_examples():
    """Verify solver behaves on a few hand-built levels."""
    from platformer_game import PlatLevel
    import numpy as np
    EMPTY, SOLID, SPIKE = 0, 1, 2

    # Trivial flat level: 5 cols of ground, no obstacles
    t = np.zeros((6, 8), dtype=np.uint8)
    t[5, :] = SOLID
    res = solve(PlatLevel(t))
    assert res.solved, "flat level should solve"
    print(f"  flat (8 cols): solved, plan_len={res.plan_len} jumps={res.n_jumps}")

    # Level with a single pit you must jump
    t2 = np.zeros((6, 8), dtype=np.uint8)
    t2[5, :] = SOLID
    t2[5, 3] = EMPTY   # pit at col 3
    res2 = solve(PlatLevel(t2))
    assert res2.solved, "single-pit level should solve"
    print(f"  pit at col 3:  solved, plan_len={res2.plan_len} jumps={res2.n_jumps}")

    # Unsolvable: huge gap (no way to jump 5 cols)
    t3 = np.zeros((6, 12), dtype=np.uint8)
    t3[5, :] = SOLID
    t3[5, 3:9] = EMPTY  # 6-wide pit, unreachable
    res3 = solve(PlatLevel(t3))
    assert not res3.solved, "wide pit should be unsolvable"
    print(f"  6-col pit:     unsolvable, nodes={res3.nodes_expanded}")


def test_random_levels(n=20):
    """Sample random levels, count solvable ratio, check timing."""
    dom = PlatformerDomain(H=10, W=24, node_budget=4000)
    rng = random.Random(42)
    t0 = time.time()
    n_solved = 0
    for _ in range(n):
        lvl = dom.init_genome(rng)
        ev = dom.evaluate(lvl)
        if ev.solvable:
            n_solved += 1
    dt = time.time() - t0
    print(f"  {n_solved}/{n} random levels solvable, {dt/n*1000:.1f} ms each")


def smoke_qd():
    dom = PlatformerDomain(H=10, W=24, node_budget=4000)
    common = DESCRIPTORS["COMMON"]
    n_evals = 600
    print(f"{'cond':<8} {'cov':>4} {'qd':>6} {'maxf':>6} {'t(s)':>6}")
    for cond in ["RAND", "STR", "PLAY", "SKILL", "COMMON"]:
        t0 = time.time()
        if cond == "RAND":
            arch, log = run_random_search(dom, common, n_evals=n_evals, seed=0)
            shown = arch
        else:
            arch, com, log = run_map_elites(
                dom, DESCRIPTORS[cond], common,
                n_evals=n_evals, n_init=60, seed=0,
            )
            shown = com
        print(f"{cond:<8} {shown.coverage():>4} {shown.qd_score():>6.2f} "
              f"{shown.max_fitness():>6.3f} {time.time()-t0:>6.1f}")


if __name__ == "__main__":
    print("--- solver examples ---")
    test_solver_examples()
    print("--- random levels ---")
    test_random_levels()
    print("--- QD smoke ---")
    smoke_qd()
