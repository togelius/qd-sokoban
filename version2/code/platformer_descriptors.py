"""Evaluation, fitness, and behavioural descriptors for the platformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from platformer_game import PlatLevel, SOLID, SPIKE
from platformer_solver import solve, PlatSolveResult


TRIVIAL_PLAN_THRESHOLD = 4
LENGTH_REWARD_CAP = 40
JUMP_REWARD_CAP = 12


@dataclass
class PlatEvalResult:
    well_formed: bool
    solvable: bool
    fitness: float
    plan_len: int
    n_jumps: int
    nodes_expanded: int
    timed_out: bool
    solver_called: bool
    n_pits: int
    height_var_q: int          # quantized stddev of column ground heights
    height_std: float
    n_spikes: int


def _count_pits(level: PlatLevel) -> int:
    """A pit is a column with no SOLID tile anywhere."""
    H, W = level.tiles.shape
    return int(sum(1 for c in range(W)
                   if not np.any(level.tiles[:, c] == SOLID)))


def _ground_heights(level: PlatLevel) -> np.ndarray:
    """Per-column ground height (number of contiguous SOLID tiles at
    bottom). Zero for pit columns."""
    H, W = level.tiles.shape
    out = np.zeros(W, dtype=int)
    for c in range(W):
        h = 0
        for r in range(H - 1, -1, -1):
            if level.tiles[r, c] == SOLID:
                h += 1
            else:
                break
        out[c] = h
    return out


def _count_spikes(level: PlatLevel) -> int:
    return int(np.sum(level.tiles == SPIKE))


def evaluate(level: PlatLevel, node_budget: int = 4000) -> PlatEvalResult:
    # quick well-formedness: must have a starting standing position
    start = level.starting_state()
    well_formed = start is not None
    n_pits = _count_pits(level)
    heights = _ground_heights(level)
    h_std = float(np.std(heights))
    # quantize height variability into 10 bins
    hv_q = int(np.clip(np.digitize(h_std,
                                   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 100.0]), 0, 9))
    n_spikes = _count_spikes(level)

    if not well_formed:
        return PlatEvalResult(False, False, 0.0, 0, 0, 0, False, False,
                              n_pits, hv_q, h_std, n_spikes)

    res: PlatSolveResult = solve(level, node_budget=node_budget)
    if not res.solved or res.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return PlatEvalResult(True, res.solved, 0.0,
                              res.plan_len, res.n_jumps,
                              res.nodes_expanded, res.timed_out, True,
                              n_pits, hv_q, h_std, n_spikes)

    # fitness composition
    length_term = min(res.plan_len, LENGTH_REWARD_CAP) / LENGTH_REWARD_CAP
    jump_term = min(res.n_jumps, JUMP_REWARD_CAP) / JUMP_REWARD_CAP
    # structural bonus rewards moderate complexity
    pit_term = max(0.0, 1.0 - abs(n_pits - 3) / 5.0)
    spike_term = max(0.0, 1.0 - abs(n_spikes - 2) / 4.0)
    fit = 0.20 + 0.40 * length_term + 0.20 * jump_term + 0.10 * pit_term + 0.10 * spike_term

    return PlatEvalResult(True, True, fit,
                          res.plan_len, res.n_jumps,
                          res.nodes_expanded, res.timed_out, True,
                          n_pits, hv_q, h_std, n_spikes)


# ---------------------------------------------------------------------------
# Descriptor bin edges
# ---------------------------------------------------------------------------

#
# Bin edges chosen from pilot data on (H=10, W=24, node_budget=4000).
# Observed ranges across ME-COMMON elites at this size:
#   plan_len in [39, 55]; n_jumps in [7, 10]; log_nodes in [1.3, 1.5];
#   n_pits in [0, 12]; height_std in [0, 3.5].
# We use bins that put the median in the middle of the range and don't
# allocate cells to combinations that are infeasible by physics.
#
STR_PIT_BINS = [0, 1, 2, 3, 4, 5, 6, 8, 100]                    # 8 bins
STR_SHAPE = (8, 10)                                              # pits x height-var

PLAY_PLAN_BINS = [4, 22, 30, 36, 42, 46, 50, 54, 60, 80, 10000] # 10 bins
PLAY_JUMP_BINS = [0, 1, 2, 4, 6, 8, 10, 12, 15, 20, 100]        # 10 bins
PLAY_SHAPE = (10, 10)

SKILL_LEN_BINS = PLAY_PLAN_BINS
SKILL_LOGN_BINS = np.linspace(1.0, 2.5, 11)                     # log10 nodes
SKILL_SHAPE = (10, 10)

COMMON_PIT_BINS = STR_PIT_BINS
COMMON_LEN_BINS = PLAY_PLAN_BINS
COMMON_SHAPE = (8, 10)


def str_cell(level: PlatLevel, ev: PlatEvalResult) -> Optional[Tuple[int, int]]:
    if not ev.well_formed:
        return None
    i = int(np.clip(np.digitize(ev.n_pits, STR_PIT_BINS, right=False) - 1, 0, STR_SHAPE[0] - 1))
    j = ev.height_var_q
    return (i, j)


def play_cell(level: PlatLevel, ev: PlatEvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    i = int(np.clip(np.digitize(ev.plan_len, PLAY_PLAN_BINS, right=False) - 1, 0, 9))
    j = int(np.clip(np.digitize(ev.n_jumps, PLAY_JUMP_BINS, right=False) - 1, 0, 9))
    return (i, j)


def skill_cell(level: PlatLevel, ev: PlatEvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    i = int(np.clip(np.digitize(ev.plan_len, SKILL_LEN_BINS, right=False) - 1, 0, 9))
    logn = math.log10(max(ev.nodes_expanded, 1))
    j = int(np.clip(np.digitize(logn, SKILL_LOGN_BINS) - 1, 0, 9))
    return (i, j)


def common_cell(level: PlatLevel, ev: PlatEvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    i = int(np.clip(np.digitize(ev.n_pits, COMMON_PIT_BINS, right=False) - 1, 0, COMMON_SHAPE[0] - 1))
    j = int(np.clip(np.digitize(ev.plan_len, COMMON_LEN_BINS, right=False) - 1, 0, 9))
    return (i, j)
