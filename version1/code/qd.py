"""MAP-Elites and a random-search baseline."""

from __future__ import annotations

import json
import math
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from sokoban import Level
from generator import random_level, mutate
from descriptors import EvalCache, fitness


@dataclass
class Cell:
    level: Level
    fit: float
    cache: EvalCache  # carry the cache so common projection is free


@dataclass
class Archive:
    """A 2-D MAP-Elites archive (with optional descriptor name)."""
    shape: Tuple[int, int]
    name: str
    grid: Dict[Tuple[int, int], Cell] = field(default_factory=dict)

    def maybe_insert(self, cell_idx, level, fit, cache) -> bool:
        if cell_idx is None or fit <= 0:
            return False
        prev = self.grid.get(cell_idx)
        if prev is None or fit > prev.fit:
            self.grid[cell_idx] = Cell(level, fit, cache)
            return True
        return False

    def coverage(self) -> int:
        return len(self.grid)

    def qd_score(self) -> float:
        return sum(c.fit for c in self.grid.values())

    def max_fit(self) -> float:
        if not self.grid:
            return 0.0
        return max(c.fit for c in self.grid.values())

    def mean_fit(self) -> float:
        if not self.grid:
            return 0.0
        return sum(c.fit for c in self.grid.values()) / len(self.grid)

    def elites(self) -> List[Cell]:
        return list(self.grid.values())


@dataclass
class RunLog:
    """Per-generation metrics. `common_*` are recomputed from a side-archive
    that is updated incrementally — every individual is also inserted into a
    (4,10) archive keyed by `(n_boxes, plan_len)`. This is the metric we
    actually compare across methods."""
    gen: List[int] = field(default_factory=list)
    coverage: List[int] = field(default_factory=list)
    qd_score: List[float] = field(default_factory=list)
    max_fit: List[float] = field(default_factory=list)
    mean_fit: List[float] = field(default_factory=list)
    n_solver_calls: List[int] = field(default_factory=list)
    wall_time_s: List[float] = field(default_factory=list)
    common_coverage: List[int] = field(default_factory=list)
    common_qd_score: List[float] = field(default_factory=list)
    common_max_fit: List[float] = field(default_factory=list)
    common_mean_fit: List[float] = field(default_factory=list)


def init_population(
    archive: Archive,
    common_archive: Archive,
    descriptor_fn: Callable,
    n_init: int,
    H: int,
    W: int,
    rng: random.Random,
    node_budget: int,
) -> int:
    """Generate `n_init` random levels and try to insert. Returns solver-call count."""
    from descriptors import common_cell
    n_solver = 0
    for _ in range(n_init):
        nb = rng.choice([1, 2, 3])
        wp = rng.uniform(0.0, 0.3)
        lvl = random_level(H, W, nb, wp, rng)
        cache = EvalCache()
        f = fitness(lvl, cache, node_budget=node_budget)
        if cache.solve_result is not None:
            n_solver += 1
        archive.maybe_insert(descriptor_fn(lvl, cache), lvl, f, cache)
        common_archive.maybe_insert(common_cell(lvl, cache), lvl, f, cache)
    return n_solver


def run_map_elites(
    descriptor_fn: Callable,
    archive_shape: Tuple[int, int],
    archive_name: str,
    n_generations: int,
    n_init: int,
    H: int,
    W: int,
    seed: int,
    node_budget: int = 5000,
    log_every: int = 50,
) -> Tuple[Archive, RunLog, Archive]:
    """Run MAP-Elites for `n_generations` selection cycles. Returns
    (own_archive, run_log, common_archive)."""
    from descriptors import common_cell, COMMON_SHAPE
    rng = random.Random(seed)
    archive = Archive(shape=archive_shape, name=archive_name)
    common = Archive(shape=COMMON_SHAPE, name=f"{archive_name}__COMMON")
    log = RunLog()
    t0 = time.time()
    n_solver = init_population(
        archive, common, descriptor_fn, n_init, H, W, rng, node_budget
    )

    for gen in range(1, n_generations + 1):
        elites = archive.elites()
        if not elites:
            nb = rng.choice([1, 2, 3])
            wp = rng.uniform(0.0, 0.3)
            parent = random_level(H, W, nb, wp, rng)
        else:
            parent = rng.choice(elites).level
        child = mutate(parent, rng)
        cache = EvalCache()
        f = fitness(child, cache, node_budget=node_budget)
        if cache.solve_result is not None:
            n_solver += 1
        archive.maybe_insert(descriptor_fn(child, cache), child, f, cache)
        common.maybe_insert(common_cell(child, cache), child, f, cache)

        if gen % log_every == 0 or gen == n_generations:
            log.gen.append(gen)
            log.coverage.append(archive.coverage())
            log.qd_score.append(archive.qd_score())
            log.max_fit.append(archive.max_fit())
            log.mean_fit.append(archive.mean_fit())
            log.n_solver_calls.append(n_solver)
            log.wall_time_s.append(time.time() - t0)
            log.common_coverage.append(common.coverage())
            log.common_qd_score.append(common.qd_score())
            log.common_max_fit.append(common.max_fit())
            log.common_mean_fit.append(common.mean_fit())
    return archive, log, common


def run_random_search(
    n_generations: int,
    n_init: int,
    H: int,
    W: int,
    seed: int,
    node_budget: int = 5000,
    log_every: int = 50,
):
    """Random search baseline. Stores the best-fitness level per
    (n_boxes, plan_len) cell of the common descriptor."""
    from descriptors import common_cell, COMMON_SHAPE
    rng = random.Random(seed)
    archive = Archive(shape=COMMON_SHAPE, name="RAND")
    log = RunLog()
    t0 = time.time()
    n_solver = 0

    total = n_init + n_generations
    for it in range(1, total + 1):
        nb = rng.choice([1, 2, 3])
        wp = rng.uniform(0.0, 0.3)
        lvl = random_level(H, W, nb, wp, rng)
        cache = EvalCache()
        f = fitness(lvl, cache, node_budget=node_budget)
        if cache.solve_result is not None:
            n_solver += 1
        archive.maybe_insert(common_cell(lvl, cache), lvl, f, cache)
        if it % log_every == 0 or it == total:
            log.gen.append(it)
            log.coverage.append(archive.coverage())
            log.qd_score.append(archive.qd_score())
            log.max_fit.append(archive.max_fit())
            log.mean_fit.append(archive.mean_fit())
            log.n_solver_calls.append(n_solver)
            log.wall_time_s.append(time.time() - t0)
            # For random search, "common" archive IS the archive
            log.common_coverage.append(archive.coverage())
            log.common_qd_score.append(archive.qd_score())
            log.common_max_fit.append(archive.max_fit())
            log.common_mean_fit.append(archive.mean_fit())
    return archive, log


# --------------------------------------------------------------------
# Projection onto the COMMON descriptor for fair cross-method comparison
# --------------------------------------------------------------------

def project_to_common(archive: Archive) -> Archive:
    """Re-insert all cells into a fresh archive indexed by the common
    descriptor (n_boxes, plan_len). Reuses cached solve_results."""
    from descriptors import common_cell, COMMON_SHAPE
    out = Archive(shape=COMMON_SHAPE, name=f"{archive.name}__COMMON")
    for cell in archive.elites():
        idx = common_cell(cell.level, cell.cache)
        out.maybe_insert(idx, cell.level, cell.fit, cell.cache)
    return out


# --------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------

def archive_to_dict(archive: Archive) -> dict:
    return {
        "shape": list(archive.shape),
        "name": archive.name,
        "cells": [
            {
                "idx": list(idx),
                "fit": cell.fit,
                "render": cell.level.render(),
                "n_boxes": len(cell.level.box_start),
                "plan_len": (cell.cache.solve_result.plan_len
                             if cell.cache.solve_result and cell.cache.solve_result.solved else None),
                "n_pushes": (cell.cache.solve_result.n_pushes
                             if cell.cache.solve_result and cell.cache.solve_result.solved else None),
                "nodes_expanded": (cell.cache.solve_result.nodes_expanded
                                   if cell.cache.solve_result else None),
                "wall_density": cell.cache.wall_density,
            }
            for idx, cell in archive.grid.items()
        ],
    }


def log_to_dict(log: RunLog) -> dict:
    return dict(
        gen=log.gen,
        coverage=log.coverage,
        qd_score=log.qd_score,
        max_fit=log.max_fit,
        mean_fit=log.mean_fit,
        n_solver_calls=log.n_solver_calls,
        wall_time_s=log.wall_time_s,
        common_coverage=log.common_coverage,
        common_qd_score=log.common_qd_score,
        common_max_fit=log.common_max_fit,
        common_mean_fit=log.common_mean_fit,
    )
