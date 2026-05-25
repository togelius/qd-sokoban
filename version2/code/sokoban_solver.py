"""A* solver for Sokoban with deadlock pruning."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from sokoban_game import Level, State


@dataclass
class SolveResult:
    solved: bool
    plan_len: int = 0
    n_pushes: int = 0
    nodes_expanded: int = 0
    timed_out: bool = False


def _heuristic(state: State, level: Level) -> int:
    h = 0
    goals = level.goals
    for (br, bc) in state.boxes:
        if (br, bc) in goals:
            continue
        best = None
        for (gr, gc) in goals:
            d = abs(br - gr) + abs(bc - gc)
            if best is None or d < best:
                best = d
        h += best
    return h


def _has_deadlock(state: State, dead_cells) -> bool:
    for b in state.boxes:
        if dead_cells[b]:
            return True
    return False


def solve(level: Level, node_budget: int = 8000) -> SolveResult:
    start = level.initial_state()
    if level.is_solved(start):
        return SolveResult(solved=True, plan_len=0, n_pushes=0, nodes_expanded=0)
    dead = level.deadlock_cells()
    if _has_deadlock(start, dead):
        return SolveResult(solved=False)

    counter = 0
    h0 = _heuristic(start, level)
    open_heap = [(h0, 0, counter, start, 0)]
    best_g = {start: 0}
    expanded = 0

    while open_heap:
        f, g, _, state, npush = heapq.heappop(open_heap)
        if g > best_g.get(state, g):
            continue
        if level.is_solved(state):
            return SolveResult(solved=True, plan_len=g, n_pushes=npush,
                               nodes_expanded=expanded)
        expanded += 1
        if expanded > node_budget:
            return SolveResult(solved=False, nodes_expanded=expanded, timed_out=True)
        for nxt, _action, pushed in level.successors(state):
            if pushed and _has_deadlock(nxt, dead):
                continue
            ng = g + 1
            if ng < best_g.get(nxt, 10**9):
                best_g[nxt] = ng
                nh = _heuristic(nxt, level)
                counter += 1
                heapq.heappush(open_heap,
                               (ng + nh, ng, counter, nxt,
                                npush + (1 if pushed else 0)))
    return SolveResult(solved=False, nodes_expanded=expanded)
