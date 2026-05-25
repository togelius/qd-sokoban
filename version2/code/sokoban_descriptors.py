"""Evaluation, fitness, and behavioural descriptors for Sokoban (v2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sokoban_game import Level
from sokoban_solver import solve, SolveResult


TRIVIAL_PLAN_THRESHOLD = 4
LENGTH_REWARD_CAP = 40
PUSH_REWARD_CAP = 20


@dataclass
class EvalResult:
    well_formed: bool
    solvable: bool
    fitness: float
    plan_len: int
    n_pushes: int
    nodes_expanded: int
    timed_out: bool
    solver_called: bool
    n_boxes: int
    wall_density: float


def evaluate(level: Level, node_budget: int = 8000) -> EvalResult:
    well_formed = level.is_well_formed()
    n_boxes = level.n_boxes()
    wd = level.wall_density()

    if not well_formed:
        return EvalResult(False, False, 0.0, 0, 0, 0, False, False, n_boxes, wd)

    res: SolveResult = solve(level, node_budget=node_budget)
    if not res.solved or res.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return EvalResult(True, res.solved, 0.0,
                          res.plan_len, res.n_pushes,
                          res.nodes_expanded, res.timed_out, True,
                          n_boxes, wd)

    length_term = min(res.plan_len, LENGTH_REWARD_CAP) / LENGTH_REWARD_CAP
    push_term = min(res.n_pushes, PUSH_REWARD_CAP) / PUSH_REWARD_CAP
    struct_bonus = max(0.0, 1.0 - abs(wd - 0.2) / 0.3)
    fit = 0.25 + 0.40 * length_term + 0.25 * push_term + 0.10 * struct_bonus

    return EvalResult(True, True, fit,
                      res.plan_len, res.n_pushes,
                      res.nodes_expanded, res.timed_out, True,
                      n_boxes, wd)


# Bin edges
STR_DENSITY_BINS = np.linspace(0.0, 0.5, 11)
STR_SHAPE = (4, 10)

PLAY_PLAN_BINS = [4, 6, 8, 10, 14, 18, 22, 28, 36, 50, 1000]
PLAY_PUSH_BINS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 1000]
PLAY_SHAPE = (10, 10)

SKILL_LEN_BINS = PLAY_PLAN_BINS
SKILL_LOGN_BINS = np.linspace(0.0, 4.0, 11)
SKILL_SHAPE = (10, 10)

COMMON_LEN_BINS = PLAY_PLAN_BINS
COMMON_SHAPE = (4, 10)


def str_cell(level: Level, ev: EvalResult) -> Optional[Tuple[int, int]]:
    if not ev.well_formed:
        return None
    if ev.n_boxes < 1 or ev.n_boxes > 4:
        return None
    j = int(np.clip(np.digitize(ev.wall_density, STR_DENSITY_BINS) - 1, 0, 9))
    return (ev.n_boxes - 1, j)


def play_cell(level: Level, ev: EvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    i = int(np.clip(np.digitize(ev.plan_len, PLAY_PLAN_BINS, right=False) - 1, 0, 9))
    j = int(np.clip(np.digitize(ev.n_pushes, PLAY_PUSH_BINS, right=False) - 1, 0, 9))
    return (i, j)


def skill_cell(level: Level, ev: EvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    i = int(np.clip(np.digitize(ev.plan_len, SKILL_LEN_BINS, right=False) - 1, 0, 9))
    logn = math.log10(max(ev.nodes_expanded, 1))
    j = int(np.clip(np.digitize(logn, SKILL_LOGN_BINS) - 1, 0, 9))
    return (i, j)


def common_cell(level: Level, ev: EvalResult) -> Optional[Tuple[int, int]]:
    if not ev.solvable or ev.plan_len < TRIVIAL_PLAN_THRESHOLD:
        return None
    if ev.n_boxes < 1 or ev.n_boxes > 4:
        return None
    j = int(np.clip(np.digitize(ev.plan_len, COMMON_LEN_BINS, right=False) - 1, 0, 9))
    return (ev.n_boxes - 1, j)
