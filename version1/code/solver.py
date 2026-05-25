"""A* solver for Sokoban with deadlock pruning and an admissible heuristic.

The heuristic is the sum, over boxes not on goals, of the Manhattan
distance from the box to its nearest goal — admissible because each
push moves a box by exactly one cell.

`solve` returns a SolveResult with:
  - solved: bool
  - plan_len: int (number of player moves, including non-push moves)
  - n_pushes: int (number of moves in the plan that pushed a box)
  - nodes_expanded: int
  - timed_out: bool (True if node budget was exhausted)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from sokoban import Level, State, DIRS


@dataclass
class SolveResult:
    solved: bool
    plan_len: int = 0
    n_pushes: int = 0
    nodes_expanded: int = 0
    timed_out: bool = False


def heuristic(state: State, level: Level) -> int:
    """Sum over unplaced boxes of Manhattan distance to nearest goal."""
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


def has_simple_deadlock(state: State, dead_cells) -> bool:
    """True if any box is on a known-deadlock cell and not on a goal."""
    for b in state.boxes:
        if dead_cells[b]:
            return True
    return False


def solve(level: Level, node_budget: int = 5000) -> SolveResult:
    """A* on player moves. Returns SolveResult."""
    start = level.initial_state()
    if level.is_solved(start):
        return SolveResult(solved=True, plan_len=0, n_pushes=0, nodes_expanded=0)

    dead = level.deadlock_cells()
    if has_simple_deadlock(start, dead):
        return SolveResult(solved=False)

    # Open list: (f, g, counter, state, n_pushes)
    counter = 0
    h0 = heuristic(start, level)
    open_heap = [(h0, 0, counter, start, 0)]
    best_g = {start: 0}
    expanded = 0

    while open_heap:
        f, g, _, state, npush = heapq.heappop(open_heap)
        if g > best_g.get(state, g):
            continue
        if level.is_solved(state):
            return SolveResult(
                solved=True,
                plan_len=g,
                n_pushes=npush,
                nodes_expanded=expanded,
            )

        expanded += 1
        if expanded > node_budget:
            return SolveResult(
                solved=False,
                nodes_expanded=expanded,
                timed_out=True,
            )

        for nxt, _action, pushed in level.successors(state):
            if pushed and has_simple_deadlock(nxt, dead):
                continue
            ng = g + 1
            if ng < best_g.get(nxt, 10**9):
                best_g[nxt] = ng
                nh = heuristic(nxt, level)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (ng + nh, ng, counter, nxt, npush + (1 if pushed else 0)),
                )

    return SolveResult(solved=False, nodes_expanded=expanded)
