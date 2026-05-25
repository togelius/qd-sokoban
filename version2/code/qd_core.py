"""Domain-agnostic Quality-Diversity primitives.

A *domain* is anything that satisfies the loose protocol used here:

    init_genome(rng) -> genome
    mutate(parent, rng) -> child
    evaluate(genome) -> EvalResult
    descriptors: Dict[str, Descriptor]

`EvalResult` is whatever the domain returns from `evaluate`; the only
contract is that it has a `.fitness: float` attribute and that the
descriptor functions can read from it. We pass the same `EvalResult`
to all descriptors so the (expensive) solver call is shared.

`Descriptor` is a dataclass:
    name, shape (Tuple[int,int]), to_cell(genome, eval_result) -> Optional[(i,j)]

This file knows nothing about Sokoban or platformers — both domains
plug into the same code path.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Descriptor:
    name: str
    shape: Tuple[int, int]
    to_cell: Callable[[Any, Any], Optional[Tuple[int, int]]]
    axis_labels: Tuple[str, str] = ("x", "y")


@dataclass
class Cell:
    genome: Any
    fitness: float
    eval_result: Any


@dataclass
class Archive:
    """2-D MAP-Elites archive."""
    shape: Tuple[int, int]
    name: str
    grid: Dict[Tuple[int, int], Cell] = field(default_factory=dict)

    def maybe_insert(self, idx: Optional[Tuple[int, int]],
                     genome: Any, fitness: float, eval_result: Any) -> bool:
        if idx is None or fitness <= 0:
            return False
        prev = self.grid.get(idx)
        if prev is None or fitness > prev.fitness:
            self.grid[idx] = Cell(genome, fitness, eval_result)
            return True
        return False

    def coverage(self) -> int:
        return len(self.grid)

    def qd_score(self) -> float:
        return sum(c.fitness for c in self.grid.values())

    def max_fitness(self) -> float:
        if not self.grid:
            return 0.0
        return max(c.fitness for c in self.grid.values())

    def mean_fitness(self) -> float:
        if not self.grid:
            return 0.0
        return sum(c.fitness for c in self.grid.values()) / len(self.grid)

    def cells(self) -> List[Cell]:
        return list(self.grid.values())


@dataclass
class RunLog:
    """Per-log-tick metrics, sampled every `log_every` evaluations."""
    eval_count: List[int] = field(default_factory=list)
    coverage: List[int] = field(default_factory=list)
    qd_score: List[float] = field(default_factory=list)
    max_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    solver_calls: List[int] = field(default_factory=list)
    wall_time_s: List[float] = field(default_factory=list)
    # COMMON-projection metrics, used for cross-method comparison.
    common_coverage: List[int] = field(default_factory=list)
    common_qd_score: List[float] = field(default_factory=list)
    common_max_fitness: List[float] = field(default_factory=list)
    common_mean_fitness: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MAP-Elites and Random-Search loops
# ---------------------------------------------------------------------------

def run_map_elites(
    domain,
    descriptor: Descriptor,
    common: Descriptor,
    n_evals: int,
    n_init: int,
    seed: int,
    log_every: int = 200,
    eval_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Archive, Archive, RunLog]:
    """Run MAP-Elites for `n_evals` evaluations (counting initialisation).

    Returns (descriptor_archive, common_archive, log).

    If `eval_records` is provided, every individual's per-evaluation
    summary is appended to it (used for the per-evaluation mechanism
    analysis).
    """
    rng = random.Random(seed)
    archive = Archive(shape=descriptor.shape, name=descriptor.name)
    common_archive = Archive(shape=common.shape, name=f"{descriptor.name}__COMMON")
    log = RunLog()
    t0 = time.time()
    solver_calls = 0

    def insert_and_record(genome, ev, idx_active, idx_common):
        nonlocal solver_calls
        solver_called = getattr(ev, "solver_called", False)
        if solver_called:
            solver_calls += 1
        archive.maybe_insert(idx_active, genome, ev.fitness, ev)
        common_archive.maybe_insert(idx_common, genome, ev.fitness, ev)
        if eval_records is not None:
            eval_records.append(_record(genome, ev, idx_active, idx_common))

    def tick(it):
        log.eval_count.append(it)
        log.coverage.append(archive.coverage())
        log.qd_score.append(archive.qd_score())
        log.max_fitness.append(archive.max_fitness())
        log.mean_fitness.append(archive.mean_fitness())
        log.solver_calls.append(solver_calls)
        log.wall_time_s.append(time.time() - t0)
        log.common_coverage.append(common_archive.coverage())
        log.common_qd_score.append(common_archive.qd_score())
        log.common_max_fitness.append(common_archive.max_fitness())
        log.common_mean_fitness.append(common_archive.mean_fitness())

    it = 0
    # ---- initial random population ----
    for _ in range(n_init):
        g = domain.init_genome(rng)
        ev = domain.evaluate(g)
        idx_a = descriptor.to_cell(g, ev)
        idx_c = common.to_cell(g, ev)
        insert_and_record(g, ev, idx_a, idx_c)
        it += 1
        if it % log_every == 0:
            tick(it)

    # ---- main loop ----
    while it < n_evals:
        elites = archive.cells()
        if not elites:
            parent_g = domain.init_genome(rng)
        else:
            parent_g = rng.choice(elites).genome
        child = domain.mutate(parent_g, rng)
        ev = domain.evaluate(child)
        idx_a = descriptor.to_cell(child, ev)
        idx_c = common.to_cell(child, ev)
        insert_and_record(child, ev, idx_a, idx_c)
        it += 1
        if it % log_every == 0:
            tick(it)

    if log.eval_count[-1] != it:
        tick(it)
    return archive, common_archive, log


def run_random_search(
    domain,
    common: Descriptor,
    n_evals: int,
    seed: int,
    log_every: int = 200,
    eval_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Archive, RunLog]:
    """Random-search baseline. The archive itself is keyed by the COMMON
    descriptor — no separate own-archive, since the baseline has no
    behavioural descriptor of its own."""
    rng = random.Random(seed)
    archive = Archive(shape=common.shape, name="RAND")
    log = RunLog()
    t0 = time.time()
    solver_calls = 0

    for it in range(1, n_evals + 1):
        g = domain.init_genome(rng)
        ev = domain.evaluate(g)
        if getattr(ev, "solver_called", False):
            solver_calls += 1
        idx_c = common.to_cell(g, ev)
        archive.maybe_insert(idx_c, g, ev.fitness, ev)
        if eval_records is not None:
            eval_records.append(_record(g, ev, idx_c, idx_c))
        if it % log_every == 0:
            log.eval_count.append(it)
            log.coverage.append(archive.coverage())
            log.qd_score.append(archive.qd_score())
            log.max_fitness.append(archive.max_fitness())
            log.mean_fitness.append(archive.mean_fitness())
            log.solver_calls.append(solver_calls)
            log.wall_time_s.append(time.time() - t0)
            # For RAND, own == common archive
            log.common_coverage.append(archive.coverage())
            log.common_qd_score.append(archive.qd_score())
            log.common_max_fitness.append(archive.max_fitness())
            log.common_mean_fitness.append(archive.mean_fitness())
    return archive, log


# ---------------------------------------------------------------------------
# Per-evaluation record
# ---------------------------------------------------------------------------

def _record(genome, ev, idx_active, idx_common) -> Dict[str, Any]:
    """Compact dict describing one evaluation, used for mechanism analysis."""
    return dict(
        fitness=float(ev.fitness),
        solvable=bool(getattr(ev, "solvable", False)),
        plan_len=int(getattr(ev, "plan_len", 0) or 0),
        nodes_expanded=int(getattr(ev, "nodes_expanded", 0) or 0),
        descriptor_cell=tuple(idx_active) if idx_active is not None else None,
        common_cell=tuple(idx_common) if idx_common is not None else None,
        well_formed=bool(getattr(ev, "well_formed", False)),
    )
