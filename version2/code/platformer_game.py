"""Tile-based platformer level + reachable-landing relation.

Level grid: numpy uint8 array, shape (H, W). Tile alphabet:

    0 EMPTY  '.'
    1 SOLID  'X'  (ground, walls, platforms — same tile)
    2 SPIKE  'S'  (instant-death; cannot be stood on or passed through)

Goal: reach the rightmost column alive. Player starts at column 0 on
the topmost solid tile of that column (if any).

Physics — "standing-to-standing" graph approximation:

  Player occupies one tile. From a *standing position* (r, c), where
  the tile (r, c) is empty and (r+1, c) is SOLID (or r == H-1 meaning
  the player is on the bottom edge which is treated as solid floor),
  the agent can take one move per action:

  - WALK_FLAT: to (r, c+1) iff (r, c+1) is empty and (r+1, c+1) is solid.
  - WALK_OFF: to (r', c+1) where r' is the smallest row > r with
    (r', c+1) empty and (r'+1, c+1) solid, and all (r''+ < r') are
    empty in column c+1. Falling off the bottom is invalid.
  - JUMP(dx, dy): with dx in 0..MAX_DX, dy in 1..MAX_DY:
      arc-up: tiles (r-1, c), (r-2, c), ..., (r-dy, c) all empty
      traverse: tiles (r-dy, c+1), ..., (r-dy, c+dx) all empty
      then fall in column c+dx starting from row r-dy down: land at
      (r'', c+dx) where (r''+1, c+dx) solid, (r'', c+dx) empty, and
      everything between r-dy and r''-1 in column c+dx is empty.
      Spike tiles invalidate the move.

  This approximation is standard in the Mario-AI / tile-grid PCG
  literature; we adopt it because it keeps the reachability graph
  finite and tractable.

Reaching any standing position at column W-1 is a "win." A move that
would land in a spike, fall off the bottom, or pass through a spike
is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# Tile codes
EMPTY = 0
SOLID = 1
SPIKE = 2

# Physics constants
MAX_DX = 4   # max horizontal distance per jump
MAX_DY = 3   # max vertical rise per jump


@dataclass(frozen=True)
class PlatState:
    row: int
    col: int

    def __hash__(self) -> int:
        return hash((self.row, self.col))


class PlatLevel:
    """Immutable tile-grid level."""

    __slots__ = ("tiles", "H", "W")

    def __init__(self, tiles: np.ndarray):
        assert tiles.dtype == np.uint8
        self.tiles = tiles
        self.H, self.W = tiles.shape

    # ------------------------------------------------------------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def tile(self, r: int, c: int) -> int:
        return int(self.tiles[r, c])

    def is_solid(self, r: int, c: int) -> bool:
        """Solid tiles you can stand on. Off-bottom is *not* solid (death)."""
        if not self.in_bounds(r, c):
            return False
        return self.tiles[r, c] == SOLID

    def is_empty(self, r: int, c: int) -> bool:
        """Passable tile (empty, not spike). Off-grid = not passable."""
        if not self.in_bounds(r, c):
            return False
        return self.tiles[r, c] == EMPTY

    def is_spike(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.tiles[r, c] == SPIKE

    def has_solid_below(self, r: int, c: int) -> bool:
        """True if standing on (r,c) is supported."""
        # If we're at the bottom row, there's no floor below — treat as
        # unsupported. The bottom row is *not* a magic floor; it must
        # itself be solid for higher rows to stand on it, and the
        # bottom row's "standing position" is one above it.
        return self.in_bounds(r + 1, c) and self.tiles[r + 1, c] == SOLID

    # ------------------------------------------------------------------
    def starting_state(self) -> Optional[PlatState]:
        """Topmost standing position in column 0. Returns None if no
        valid start exists."""
        for r in range(self.H - 1):
            if self.tiles[r, 0] == EMPTY and self.tiles[r + 1, 0] == SOLID:
                return PlatState(r, 0)
        return None

    def is_goal(self, st: PlatState) -> bool:
        return st.col == self.W - 1

    # ------------------------------------------------------------------
    def successors(self, st: PlatState) -> List[Tuple[PlatState, int, int, str]]:
        """Return (next_state, was_jump, cost_ticks, action_name) list.

        Cost is in "game ticks" so that walking on flat ground is the
        cheapest way to advance one column. Jumping is only optimal
        when it bypasses an obstacle. This is what gives the
        `n_jumps` PLAY descriptor any spread — otherwise the optimal
        agent would jump everywhere because each jump advances up to
        MAX_DX columns "for free."

        Cost model (1 tick per row or column moved):
          - WALK:      1
          - WALK_OFF:  1 + fall_distance
          - JUMP(dx, dy): dy (rise) + dx (traverse) + fall_distance
        """
        out: List[Tuple[PlatState, int, int, str]] = []
        r, c = st.row, st.col
        # WALK_FLAT
        if self.is_empty(r, c + 1) and self.has_solid_below(r, c + 1):
            out.append((PlatState(r, c + 1), 0, 1, "walk"))
        # WALK_OFF: step into c+1, fall to first solid floor.
        if self.is_empty(r, c + 1) and not self.has_solid_below(r, c + 1):
            landed = self._fall_in_column(r, c + 1)
            if landed is not None:
                fall = landed - r
                out.append((PlatState(landed, c + 1), 0, 1 + fall, "walk_off"))
        # JUMP(dx, dy)
        for dy in range(1, MAX_DY + 1):
            if not all(self.is_empty(r - k, c) for k in range(1, dy + 1)):
                continue
            for dx in range(1, MAX_DX + 1):
                ok = True
                for k in range(1, dx + 1):
                    if not self.is_empty(r - dy, c + k):
                        ok = False
                        break
                if not ok:
                    continue
                target_col = c + dx
                landed = self._fall_in_column(r - dy, target_col,
                                              start_can_be_solid=False)
                if landed is None:
                    continue
                fall = landed - (r - dy)
                cost = dy + dx + fall
                out.append((PlatState(landed, target_col), 1, cost,
                            f"jump_{dx}_{dy}"))
        return out

    def _fall_in_column(self, start_row: int, col: int,
                        start_can_be_solid: bool = True) -> Optional[int]:
        """Starting at (start_row, col), fall until landing on a tile
        with solid below. Returns the landing row, or None if the
        player falls off the bottom or hits a spike.
        """
        if not self.in_bounds(start_row, col):
            return None
        # The starting tile must itself be empty (player needs to
        # occupy it). If start_can_be_solid is False (mid-jump landing
        # column), the tile must be empty; otherwise we're walking
        # off a cliff and the start tile is already empty by caller.
        if self.tiles[start_row, col] == SPIKE:
            return None
        if self.tiles[start_row, col] != EMPTY:
            return None
        r = start_row
        while r < self.H - 1:
            below = self.tiles[r + 1, col]
            if below == SOLID:
                return r
            if below == SPIKE:
                return None  # falling into spike = death
            # below is EMPTY; keep falling
            r += 1
        # Fell past bottom row without landing — death.
        return None

    # ------------------------------------------------------------------
    def render(self, path: Optional[List[PlatState]] = None) -> str:
        glyphs = {EMPTY: ".", SOLID: "X", SPIKE: "S"}
        grid = [[glyphs[int(self.tiles[r, c])] for c in range(self.W)]
                for r in range(self.H)]
        if path:
            for st in path:
                if grid[st.row][st.col] == ".":
                    grid[st.row][st.col] = "o"
            # mark start and end specially
            s, e = path[0], path[-1]
            grid[s.row][s.col] = "@"
            grid[e.row][e.col] = "G"
        return "\n".join("".join(row) for row in grid)
