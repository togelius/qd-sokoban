"""Sokoban game representation and mechanics.

A level is a rectangular grid with:
  - walls (immovable)
  - goal cells (positions where boxes must end up)
  - one player start position
  - one or more box start positions

A *state* is (player_position, frozenset_of_box_positions). Walls and
goals are level-level constants. The level is solved when the box set
equals the goal set.

Encoding for ASCII / serialisation:
  '#' wall, ' ' floor, '.' goal, '$' box, '*' box on goal,
  '@' player, '+' player on goal.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple


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
    """Static level data plus initial state."""

    __slots__ = (
        "walls",
        "goals",
        "player_start",
        "box_start",
        "H",
        "W",
        "_deadlock_cache",
    )

    def __init__(
        self,
        walls: np.ndarray,
        goals: FrozenSet[Pos],
        player_start: Pos,
        box_start: FrozenSet[Pos],
    ):
        assert walls.dtype == bool
        self.walls = walls
        self.goals = goals
        self.player_start = player_start
        self.box_start = box_start
        self.H, self.W = walls.shape
        self._deadlock_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State / transition
    # ------------------------------------------------------------------
    def initial_state(self) -> State:
        return State(self.player_start, self.box_start)

    def is_solved(self, state: State) -> bool:
        return state.boxes == self.goals

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def successors(self, state: State) -> List[Tuple[State, str, bool]]:
        """Return list of (next_state, action_name, pushed_a_box)."""
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def is_well_formed(self) -> bool:
        """Cheap structural sanity check (does not check solvability)."""
        if len(self.box_start) == 0:
            return False
        if len(self.goals) != len(self.box_start):
            return False
        # Player not in a wall
        if self.walls[self.player_start]:
            return False
        # Boxes not in walls, not on each other
        if any(self.walls[b] for b in self.box_start):
            return False
        # Player not on a box
        if self.player_start in self.box_start:
            return False
        # Connected free-space reachability from player
        free_reachable = self._reachable_floor(self.player_start)
        for g in self.goals:
            if g not in free_reachable:
                return False
        for b in self.box_start:
            if b not in free_reachable:
                return False
        return True

    def _reachable_floor(self, start: Pos) -> FrozenSet[Pos]:
        """Flood-fill of non-wall cells reachable from `start` (ignoring boxes)."""
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

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------
    def deadlock_cells(self) -> np.ndarray:
        """Boolean array — True for cells from which no single box can ever
        reach any goal (used as a strong static prune for the solver).

        Computed via reverse BFS: a cell is *non-deadlock* if a box on
        that cell could have been pulled (in reverse) from some goal.
        Pulls are symmetric to pushes — moving a box left requires the
        puller to be on the left.
        """
        if self._deadlock_cache is not None:
            return self._deadlock_cache

        reach = np.zeros_like(self.walls, dtype=bool)
        # Reverse BFS from each goal
        frontier: List[Pos] = []
        for g in self.goals:
            if not self.walls[g]:
                reach[g] = True
                frontier.append(g)
        while frontier:
            r, c = frontier.pop()
            for dr, dc in DIRS:
                # In reverse: box at (r,c) was pulled from (r-dr, c-dc).
                # That requires the puller stood at (r-2*dr, c-2*dc).
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

        # Walls are also "dead" trivially (we never put boxes there)
        dead = ~reach | self.walls
        self._deadlock_cache = dead
        return dead

    # ------------------------------------------------------------------
    # ASCII rendering / parsing
    # ------------------------------------------------------------------
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
                row.append("")
            lines.append("".join(row))
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> "Level":
        rows = [r for r in text.splitlines() if r.strip() != ""]
        H = len(rows)
        W = max(len(r) for r in rows)
        walls = np.zeros((H, W), dtype=bool)
        goals = set()
        boxes = set()
        player = None
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                p = (r, c)
                if ch == "#":
                    walls[r, c] = True
                elif ch == ".":
                    goals.add(p)
                elif ch == "$":
                    boxes.add(p)
                elif ch == "*":
                    boxes.add(p)
                    goals.add(p)
                elif ch == "@":
                    player = p
                elif ch == "+":
                    player = p
                    goals.add(p)
                # ' ' floor — nothing to do
            for c in range(len(line), W):
                walls[r, c] = True  # ragged rows -> pad with walls
        assert player is not None, "level has no player"
        return cls(walls, frozenset(goals), player, frozenset(boxes))
