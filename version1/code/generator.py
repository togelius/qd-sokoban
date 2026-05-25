"""Random level generators and mutation operators for direct-encoding evolution.

Genome representation: a `Genome` is just a `Level` plus convenience
methods for in-place-style transformations (we always return a new
Level, never mutate in place — Levels are treated as immutable values).

We generate on a fixed-size grid with an outer wall border, which
simplifies bounds checks.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from sokoban import Level, Pos, DIRS


def random_level(
    H: int,
    W: int,
    n_boxes: int,
    wall_prob: float,
    rng: random.Random,
    max_tries: int = 50,
) -> Level:
    """Generate a random well-formed level.

    Procedure:
      1. Build outer wall border.
      2. Randomly set interior cells to walls with probability `wall_prob`.
      3. Pick `n_boxes` distinct floor cells for goals.
      4. Pick `n_boxes` distinct floor cells for boxes (disjoint from goals).
      5. Pick one remaining floor cell for the player.
      6. Verify well-formed (everything reachable from player ignoring boxes).
    Retry up to `max_tries` then fall back to an empty room.
    """
    for _ in range(max_tries):
        lvl = _try_random(H, W, n_boxes, wall_prob, rng)
        if lvl is not None and lvl.is_well_formed():
            return lvl
    return _empty_room(H, W, n_boxes, rng)


def _try_random(H, W, n_boxes, wall_prob, rng):
    walls = np.zeros((H, W), dtype=bool)
    walls[0, :] = True
    walls[H - 1, :] = True
    walls[:, 0] = True
    walls[:, W - 1] = True
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if rng.random() < wall_prob:
                walls[r, c] = True
    floors = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if not walls[r, c]]
    needed = n_boxes * 2 + 1
    if len(floors) < needed:
        return None
    rng.shuffle(floors)
    goals = frozenset(floors[:n_boxes])
    boxes = frozenset(floors[n_boxes:2 * n_boxes])
    player = floors[2 * n_boxes]
    return Level(walls, goals, player, boxes)


def _empty_room(H, W, n_boxes, rng):
    walls = np.zeros((H, W), dtype=bool)
    walls[0, :] = walls[H - 1, :] = walls[:, 0] = walls[:, W - 1] = True
    interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    rng.shuffle(interior)
    goals = frozenset(interior[:n_boxes])
    boxes = frozenset(interior[n_boxes:2 * n_boxes])
    player = interior[2 * n_boxes]
    return Level(walls, goals, player, boxes)


# --------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------

def _interior(H, W):
    return [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]


def mutate(level: Level, rng: random.Random) -> Level:
    """Return a mutated copy of `level`.

    Picks one of several operators with fixed probabilities:
      - toggle a wall in the interior
      - move a single box to a random floor cell
      - move a single goal to a random floor cell
      - move the player
      - add a box-goal pair (if room allows)
      - remove a box-goal pair (if at least 2 boxes)

    The result is *always returned* but may fail `is_well_formed`. The
    caller (the QD loop) must handle invalid offspring.
    """
    op = rng.random()
    if op < 0.45:
        return _toggle_wall(level, rng)
    elif op < 0.7:
        return _move_random_box(level, rng)
    elif op < 0.85:
        return _move_random_goal(level, rng)
    elif op < 0.93:
        return _move_player(level, rng)
    elif op < 0.97:
        return _add_box_goal_pair(level, rng)
    else:
        return _remove_box_goal_pair(level, rng)


def _toggle_wall(level: Level, rng: random.Random) -> Level:
    H, W = level.H, level.W
    interior = _interior(H, W)
    rng.shuffle(interior)
    for (r, c) in interior:
        # don't toggle if it would obliterate a box/goal/player
        if (r, c) == level.player_start:
            continue
        if (r, c) in level.box_start or (r, c) in level.goals:
            continue
        new_walls = level.walls.copy()
        new_walls[r, c] = not new_walls[r, c]
        return Level(new_walls, level.goals, level.player_start, level.box_start)
    return level


def _free_floor_cells(level: Level) -> List[Pos]:
    cells = []
    for r in range(1, level.H - 1):
        for c in range(1, level.W - 1):
            if level.walls[r, c]:
                continue
            cells.append((r, c))
    return cells


def _move_random_box(level: Level, rng: random.Random) -> Level:
    boxes = list(level.box_start)
    if not boxes:
        return level
    idx = rng.randrange(len(boxes))
    old = boxes[idx]
    free = [p for p in _free_floor_cells(level)
            if p not in level.box_start and p != level.player_start]
    if not free:
        return level
    new_pos = rng.choice(free)
    new_boxes = (level.box_start - {old}) | {new_pos}
    return Level(level.walls, level.goals, level.player_start, frozenset(new_boxes))


def _move_random_goal(level: Level, rng: random.Random) -> Level:
    goals = list(level.goals)
    if not goals:
        return level
    idx = rng.randrange(len(goals))
    old = goals[idx]
    free = [p for p in _free_floor_cells(level) if p not in level.goals]
    if not free:
        return level
    new_pos = rng.choice(free)
    new_goals = (level.goals - {old}) | {new_pos}
    return Level(level.walls, frozenset(new_goals), level.player_start, level.box_start)


def _move_player(level: Level, rng: random.Random) -> Level:
    free = [p for p in _free_floor_cells(level) if p not in level.box_start]
    if not free:
        return level
    return Level(level.walls, level.goals, rng.choice(free), level.box_start)


def _add_box_goal_pair(level: Level, rng: random.Random) -> Level:
    free_for_box = [p for p in _free_floor_cells(level)
                    if p not in level.box_start and p != level.player_start]
    free_for_goal = [p for p in _free_floor_cells(level) if p not in level.goals]
    if len(free_for_box) < 1 or len(free_for_goal) < 1:
        return level
    box = rng.choice(free_for_box)
    goal = rng.choice(free_for_goal)
    if box == goal:
        # Would create a "*" — fine in Sokoban, but boring at init.
        pass
    new_boxes = level.box_start | {box}
    new_goals = level.goals | {goal}
    return Level(level.walls, frozenset(new_goals), level.player_start, frozenset(new_boxes))


def _remove_box_goal_pair(level: Level, rng: random.Random) -> Level:
    if len(level.box_start) <= 1:
        return level
    box = rng.choice(list(level.box_start))
    goal = rng.choice(list(level.goals))
    new_boxes = level.box_start - {box}
    new_goals = level.goals - {goal}
    return Level(level.walls, frozenset(new_goals), level.player_start, frozenset(new_boxes))
