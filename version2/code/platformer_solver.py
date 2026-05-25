"""A* solver for the platformer over standing-position states."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Optional

from platformer_game import PlatLevel, PlatState


@dataclass
class PlatSolveResult:
    solved: bool
    plan_len: int = 0           # number of moves (edges) to reach goal
    n_jumps: int = 0            # number of jump moves in plan
    nodes_expanded: int = 0
    timed_out: bool = False
    max_col_reached: int = 0    # how far the search got


def _heuristic(st: PlatState, goal_col: int) -> int:
    # Each move advances at most MAX_DX = 4 columns. Manhattan-ish
    # bound — admissible because every reachable move increments col
    # by at least 1 (we skip dx==0 jumps).
    return max(0, goal_col - st.col)


def solve(level: PlatLevel, node_budget: int = 4000) -> PlatSolveResult:
    start = level.starting_state()
    if start is None:
        return PlatSolveResult(solved=False)

    if level.is_goal(start):
        return PlatSolveResult(solved=True, plan_len=0)

    goal_col = level.W - 1
    counter = 0
    open_heap = [(0, 0, counter, start, 0)]   # (f, g, ctr, state, n_jumps)
    best_g: Dict[PlatState, int] = {start: 0}
    expanded = 0
    max_col = start.col

    while open_heap:
        f, g, _, st, njumps = heapq.heappop(open_heap)
        if g > best_g.get(st, g):
            continue
        if st.col > max_col:
            max_col = st.col
        if level.is_goal(st):
            return PlatSolveResult(solved=True, plan_len=g, n_jumps=njumps,
                                   nodes_expanded=expanded,
                                   max_col_reached=st.col)
        expanded += 1
        if expanded > node_budget:
            return PlatSolveResult(solved=False, nodes_expanded=expanded,
                                   timed_out=True, max_col_reached=max_col)
        for nxt, jumped, cost, _name in level.successors(st):
            ng = g + cost
            if ng < best_g.get(nxt, 10**9):
                best_g[nxt] = ng
                counter += 1
                nh = _heuristic(nxt, goal_col)
                heapq.heappush(open_heap,
                               (ng + nh, ng, counter, nxt,
                                njumps + jumped))
    return PlatSolveResult(solved=False, nodes_expanded=expanded,
                           max_col_reached=max_col)
