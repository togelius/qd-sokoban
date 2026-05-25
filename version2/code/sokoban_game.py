"""Sokoban level + game mechanics (v2; ported from v1 with no semantic change)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple


Pos = Tuple[int, int]
DIRS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ["U", "D", "L", "R"]


@dataclass(frozen=True)
class State:
    player: Pos
    boxes: FrozenSet[Pos]

    def __hash__(self) -> int:
        return hash((self.player, self.boxes))


class Level:
    """Static level (walls, goals, player_start, box_start).
    Treated as immutable — mutations return new Level instances."""

    __slots__ = ("walls", "goals", "player_start", "box_start",
                 "H", "W", "_deadlock_cache")

    def __init__(self, walls: np.ndarray, goals: FrozenSet[Pos],
                 player_start: Pos, box_start: FrozenSet[Pos]):
        assert walls.dtype == bool
        self.walls = walls
        self.goals = goals
        self.player_start = player_start
        self.box_start = box_start
        self.H, self.W = walls.shape
        self._deadlock_cache: Optional[np.ndarray] = None

    def initial_state(self) -> State:
        return State(self.player_start, self.box_start)

    def is_solved(self, state: State) -> bool:
        return state.boxes == self.goals

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def successors(self, state: State) -> List[Tuple[State, str, bool]]:
        out: List[Tuple[State, str, bool]] = []
        pr, pc = state.player
        for (dr, dc), name in zip(DIRS, DIR_NAMES):
            nr, nc = pr + dr, pc + dc
            if not self.in_bounds(nr, nc) or self.walls[nr, nc]:
                continue
            if (nr, nc) in state.boxes:
                br, bc = nr + dr, nc + dc
                if not self.in_bounds(br, bc) or self.walls[br, bc]:
                    continue
                if (br, bc) in state.boxes:
                    continue
                new_boxes = (state.boxes - {(nr, nc)}) | {(br, bc)}
                out.append((State((nr, nc), frozenset(new_boxes)), name, True))
            else:
                out.append((State((nr, nc), state.boxes), name, False))
        return out

    def is_well_formed(self) -> bool:
        if len(self.box_start) == 0:
            return False
        if len(self.goals) != len(self.box_start):
            return False
        if self.walls[self.player_start]:
            return False
        if any(self.walls[b] for b in self.box_start):
            return False
        if self.player_start in self.box_start:
            return False
        reach = self._reachable_floor(self.player_start)
        return all(g in reach for g in self.goals) and \
               all(b in reach for b in self.box_start)

    def _reachable_floor(self, start: Pos) -> FrozenSet[Pos]:
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and not self.walls[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return frozenset(seen)

    def deadlock_cells(self) -> np.ndarray:
        if self._deadlock_cache is not None:
            return self._deadlock_cache
        reach = np.zeros_like(self.walls, dtype=bool)
        frontier: List[Pos] = []
        for g in self.goals:
            if not self.walls[g]:
                reach[g] = True
                frontier.append(g)
        while frontier:
            r, c = frontier.pop()
            for dr, dc in DIRS:
                br, bc = r - dr, c - dc
                pr, pc = r - 2 * dr, c - 2 * dc
                if not (self.in_bounds(br, bc) and self.in_bounds(pr, pc)):
                    continue
                if self.walls[br, bc] or self.walls[pr, pc]:
                    continue
                if reach[br, bc]:
                    continue
                reach[br, bc] = True
                frontier.append((br, bc))
        dead = ~reach | self.walls
        self._deadlock_cache = dead
        return dead

    def wall_density(self) -> float:
        interior = (self.H - 2) * (self.W - 2)
        return float(self.walls[1:-1, 1:-1].sum()) / max(interior, 1)

    def n_boxes(self) -> int:
        return len(self.box_start)

    def render(self, state: Optional[State] = None) -> str:
        st = state if state is not None else self.initial_state()
        lines = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                p = (r, c)
                if self.walls[r, c]:
                    row.append("#")
                else:
                    on_goal = p in self.goals
                    has_box = p in st.boxes
                    is_player = p == st.player
                    if is_player and on_goal:
                        row.append("+")
                    elif is_player:
                        row.append("@")
                    elif has_box and on_goal:
                        row.append("*")
                    elif has_box:
                        row.append("$")
                    elif on_goal:
                        row.append(".")
                    else:
                        row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)
