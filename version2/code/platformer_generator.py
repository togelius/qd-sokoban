"""Random init + mutation operators for the platformer."""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from platformer_game import PlatLevel, EMPTY, SOLID, SPIKE


def random_level(H: int, W: int, rng: random.Random,
                 mean_ground_height: float = 2.0,
                 platform_prob: float = 0.05,
                 spike_prob: float = 0.05,
                 pit_prob: float = 0.15) -> PlatLevel:
    """Generate a random tile-grid level.

    Procedure:
      - For each column c, decide a *ground height* h_c (number of
        solid tiles stacked from the bottom). Column 0 always has at
        least 1 solid tile so the start state exists.
      - With probability `pit_prob`, set h_c = 0 (pit) for c > 0.
        Pits can't be col 0 (player needs a start) or col W-1 (player
        needs to be able to stand on the goal column).
      - Scatter floating platforms above ground level with probability
        `platform_prob` per cell.
      - Scatter spikes on top of ground tiles with probability
        `spike_prob` (one spike per column, placed just above the
        ground if any).
    """
    tiles = np.zeros((H, W), dtype=np.uint8)
    # column ground heights
    heights = []
    for c in range(W):
        if c == 0 or c == W - 1:
            # Need a standable column for start (and a goal column).
            h = max(1, int(rng.gauss(mean_ground_height, 1.5)))
        elif rng.random() < pit_prob:
            h = 0
        else:
            h = max(1, int(rng.gauss(mean_ground_height, 1.5)))
        h = min(h, H - 1)
        heights.append(h)

    for c, h in enumerate(heights):
        for k in range(h):
            tiles[H - 1 - k, c] = SOLID

    # floating platforms
    for r in range(2, H - 2):
        for c in range(W):
            # don't overwrite ground stacks
            if tiles[r, c] == SOLID:
                continue
            if rng.random() < platform_prob:
                tiles[r, c] = SOLID

    # spikes — placed on top of ground tiles (so the player must jump
    # over them). One per column at most.
    for c in range(1, W - 1):
        if rng.random() < spike_prob:
            # find top of column
            top_solid = None
            for r in range(H):
                if tiles[r, c] == SOLID:
                    top_solid = r
                    break
            if top_solid is not None and top_solid - 1 >= 0:
                tiles[top_solid - 1, c] = SPIKE

    return PlatLevel(tiles)


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate(level: PlatLevel, rng: random.Random) -> PlatLevel:
    op = rng.random()
    if op < 0.30:
        return _toggle_random_tile(level, rng)
    elif op < 0.55:
        return _change_ground_height(level, rng)
    elif op < 0.70:
        return _add_or_remove_pit(level, rng)
    elif op < 0.85:
        return _toggle_spike(level, rng)
    else:
        return _toggle_floating_platform(level, rng)


def _toggle_random_tile(level: PlatLevel, rng: random.Random) -> PlatLevel:
    t = level.tiles.copy()
    H, W = t.shape
    # avoid breaking col 0 start: don't touch col 0's topmost solid
    for _ in range(20):
        r = rng.randrange(0, H - 1)
        c = rng.randrange(1, W - 1)
        cur = t[r, c]
        # cycle EMPTY <-> SOLID
        t[r, c] = SOLID if cur == EMPTY else EMPTY
        return PlatLevel(t)
    return level


def _change_ground_height(level: PlatLevel, rng: random.Random) -> PlatLevel:
    t = level.tiles.copy()
    H, W = t.shape
    c = rng.randrange(1, W - 1)
    # find current ground height
    h = 0
    for r in range(H - 1, -1, -1):
        if t[r, c] == SOLID:
            h += 1
        else:
            break
    delta = rng.choice([-2, -1, 1, 2])
    new_h = max(0, min(H - 1, h + delta))
    # rebuild column with new_h ground; clear anything above existing ground
    for r in range(H):
        if t[r, c] == SOLID and r >= H - h:
            t[r, c] = EMPTY
        # clear spike sitting on now-removed ground
        if t[r, c] == SPIKE and r >= H - h - 1:
            t[r, c] = EMPTY
    for k in range(new_h):
        t[H - 1 - k, c] = SOLID
    return PlatLevel(t)


def _add_or_remove_pit(level: PlatLevel, rng: random.Random) -> PlatLevel:
    t = level.tiles.copy()
    H, W = t.shape
    c = rng.randrange(1, W - 1)
    # is column c a pit?
    is_pit = not any(t[r, c] == SOLID for r in range(H))
    if is_pit:
        # add ground
        h = rng.randint(1, 3)
        for k in range(h):
            t[H - 1 - k, c] = SOLID
    else:
        # remove ground (turn into pit) — clear column
        for r in range(H):
            t[r, c] = EMPTY
    return PlatLevel(t)


def _toggle_spike(level: PlatLevel, rng: random.Random) -> PlatLevel:
    t = level.tiles.copy()
    H, W = t.shape
    c = rng.randrange(1, W - 1)
    # find top of column
    top_solid = None
    for r in range(H):
        if t[r, c] == SOLID:
            top_solid = r
            break
    if top_solid is None or top_solid - 1 < 0:
        return level
    above = t[top_solid - 1, c]
    if above == SPIKE:
        t[top_solid - 1, c] = EMPTY
    elif above == EMPTY:
        t[top_solid - 1, c] = SPIKE
    return PlatLevel(t)


def _toggle_floating_platform(level: PlatLevel, rng: random.Random) -> PlatLevel:
    t = level.tiles.copy()
    H, W = t.shape
    r = rng.randrange(1, H - 3)
    c = rng.randrange(1, W - 1)
    cur = t[r, c]
    if cur == SOLID:
        t[r, c] = EMPTY
    elif cur == EMPTY:
        t[r, c] = SOLID
    return PlatLevel(t)
