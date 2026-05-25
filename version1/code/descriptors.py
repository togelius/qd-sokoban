"""Behavioral descriptors and fitness functions.

A *descriptor* is a function `level -> (i, j)` mapping a level into a
2-D grid cell index (or `None` if out of bounds / not applicable).
A *fitness* is a non-negative scalar; higher is better. Infeasible
levels get fitness 0.

We pass a `Cache` around so that solver results computed once for
fitness are reused for descriptors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from sokoban import Level, State
from solver import solve, SolveResult


# ----------------------------------------------------------------------
# Evaluation cache
# ----------------------------------------------------------------------

@dataclass
class EvalCache:
    """Per-individual evaluation cache (so we never solve twice)."""
    well_formed: Optional[bool] = None
    solve_result: Optional[SolveResult] = None
    n_boxes: Optional[int] = None
    wall_density: Optional[float] = None

    def ensure_basic(self, level: Level) -> None:
        if self.well_formed is None:
            self.well_formed = level.is_well_formed()
        if self.n_boxes is None:
            self.n_boxes = len(level.box_start)
        if self.wall_density is None:
            interior = (level.H - 2) * (level.W - 2)
            wall_count = int(level.walls[1:-1, 1:-1].sum())
            self.wall_density = wall_count / max(interior, 1)

    def ensure_solve(self, level: Level, node_budget: int = 5000) -> None:
        if self.solve_result is None:
            self.solve_result = solve(level, node_budget=node_budget)


# ----------------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------------

# Minimum solution length we consider "non-trivial".
TRIVIAL_PLAN_THRESHOLD = 4
# Below this we don't reward extra length (anti-bloat).
LENGTH_REWARD_CAP = 40
# Cap for n_pushes contribution.
PUSH_REWARD_CAP = 20


def fitness(level: Level, cache: EvalCache, node_budget: int = 5000) -> float:
    """Composite fitness. Returns 0 for infeasible levels.

    fitness = solvable * (
        0.25
        + 0.4 * min(plan_len, CAP) / CAP
        + 0.25 * min(n_pushes, CAP_P) / CAP_P
        + 0.10 * structural_bonus
    )

    where structural_bonus rewards moderate wall density (penalises both
    near-empty and near-full rooms). Max fitness = 1.0.
    """
    cache.ensure_basic(level)
    if not cache.well_formed:
        return 0.0
    cache.ensure_solve(level, node_budget=node_budget)
    res = cache.solve_result
    if not res.solved:
        return 0.0
    if res.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return 0.0

    length_term = min(res.plan_len, LENGTH_REWARD_CAP) / LENGTH_REWARD_CAP
    push_term = min(res.n_pushes, PUSH_REWARD_CAP) / PUSH_REWARD_CAP
    # Structural bonus: triangular peak at density 0.2
    d = cache.wall_density
    struct = max(0.0, 1.0 - abs(d - 0.2) / 0.3)
    return 0.25 + 0.40 * length_term + 0.25 * push_term + 0.10 * struct


# ----------------------------------------------------------------------
# Behavioral descriptors
# ----------------------------------------------------------------------

#
# Each descriptor module exposes:
#   - GRID_SHAPE: (n_rows, n_cols) of the MAP-Elites archive
#   - to_cell(level, cache) -> Optional[Tuple[int,int]]
#   - axis_labels: (str, str)
#   - axis_ticks: ([label,...], [label,...])  # for plotting
#

# ---- structural (cheap, no solver call) ----

STRUCT_BOX_BINS = [1, 2, 3, 4]               # 4 bins
STRUCT_DENSITY_BINS = np.linspace(0.0, 0.5, 11)  # 10 bins (edges)
STRUCT_SHAPE = (4, 10)


def struct_cell(level: Level, cache: EvalCache) -> Optional[Tuple[int, int]]:
    cache.ensure_basic(level)
    if not cache.well_formed:
        return None
    nb = cache.n_boxes
    if nb < 1 or nb > 4:
        return None
    d = cache.wall_density
    j = int(np.clip(np.digitize(d, STRUCT_DENSITY_BINS) - 1, 0, 9))
    return (nb - 1, j)


STRUCT_DESC = dict(
    name="STR",
    shape=STRUCT_SHAPE,
    to_cell=struct_cell,
    axis_labels=("# boxes", "wall density"),
)


# ---- gameplay (expensive: requires solve) ----

PLAY_PLAN_BINS = [4, 6, 8, 10, 14, 18, 22, 28, 36, 50, 1000]  # 10 bins
PLAY_PUSH_BINS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 1000]    # 10 bins
PLAY_SHAPE = (10, 10)


def play_cell(level: Level, cache: EvalCache) -> Optional[Tuple[int, int]]:
    cache.ensure_basic(level)
    if not cache.well_formed:
        return None
    cache.ensure_solve(level)
    res = cache.solve_result
    if not res.solved:
        return None
    i = int(np.clip(np.digitize(res.plan_len, PLAY_PLAN_BINS, right=False) - 1, 0, 9))
    j = int(np.clip(np.digitize(res.n_pushes, PLAY_PUSH_BINS, right=False) - 1, 0, 9))
    return (i, j)


PLAY_DESC = dict(
    name="PLAY",
    shape=PLAY_SHAPE,
    to_cell=play_cell,
    axis_labels=("plan length", "# pushes"),
)


# ---- skill / hardness (expensive: requires solve) ----

SKILL_LEN_BINS = PLAY_PLAN_BINS                                # 10 bins
SKILL_LOGN_BINS = np.linspace(0.0, 4.0, 11)                    # 10 bins of log10(nodes)
SKILL_SHAPE = (10, 10)


def skill_cell(level: Level, cache: EvalCache) -> Optional[Tuple[int, int]]:
    cache.ensure_basic(level)
    if not cache.well_formed:
        return None
    cache.ensure_solve(level)
    res = cache.solve_result
    if not res.solved:
        return None
    i = int(np.clip(np.digitize(res.plan_len, SKILL_LEN_BINS, right=False) - 1, 0, 9))
    logn = math.log10(max(res.nodes_expanded, 1))
    j = int(np.clip(np.digitize(logn, SKILL_LOGN_BINS) - 1, 0, 9))
    return (i, j)


SKILL_DESC = dict(
    name="SKILL",
    shape=SKILL_SHAPE,
    to_cell=skill_cell,
    axis_labels=("plan length", "log10(A* nodes)"),
)


# ---- COMMON projection (used to compare archives) ----

COMMON_BOX_BINS = [1, 2, 3, 4]
COMMON_LEN_BINS = PLAY_PLAN_BINS
COMMON_SHAPE = (4, 10)


def common_cell(level: Level, cache: EvalCache) -> Optional[Tuple[int, int]]:
    cache.ensure_basic(level)
    if not cache.well_formed:
        return None
    cache.ensure_solve(level)
    res = cache.solve_result
    if not res.solved:
        return None
    nb = cache.n_boxes
    if nb < 1 or nb > 4:
        return None
    j = int(np.clip(np.digitize(res.plan_len, COMMON_LEN_BINS, right=False) - 1, 0, 9))
    return (nb - 1, j)


COMMON_DESC = dict(
    name="COMMON",
    shape=COMMON_SHAPE,
    to_cell=common_cell,
    axis_labels=("# boxes", "plan length"),
)


ALL_DESCRIPTORS = {
    "STR": STRUCT_DESC,
    "PLAY": PLAY_DESC,
    "SKILL": SKILL_DESC,
    "COMMON": COMMON_DESC,
}
