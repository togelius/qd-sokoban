"""Random level init + mutation operators for Sokoban."""

from __future__ import annotations

import random
from typing import List

import numpy as np

from sokoban_game import Level, Pos


def random_level(H: int, W: int, n_boxes: int, wall_prob: float,
                 rng: random.Random, max_tries: int = 50) -> Level:
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


def _interior(H, W) -> List[Pos]:
    return [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]


def _free_floor(level: Level) -> List[Pos]:
    return [(r, c)
            for r in range(1, level.H - 1)
            for c in range(1, level.W - 1)
            if not level.walls[r, c]]


def mutate(level: Level, rng: random.Random) -> Level:
    op = rng.random()
    if op < 0.45:
        return _toggle_wall(level, rng)
    elif op < 0.70:
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
    interior = _interior(level.H, level.W)
    rng.shuffle(interior)
    for (r, c) in interior:
        if (r, c) == level.player_start:
            continue
        if (r, c) in level.box_start or (r, c) in level.goals:
            continue
        new_walls = level.walls.copy()
        new_walls[r, c] = not new_walls[r, c]
        return Level(new_walls, level.goals, level.player_start, level.box_start)
    return level


def _move_random_box(level: Level, rng: random.Random) -> Level:
    boxes = list(level.box_start)
    if not boxes:
        return level
    old = boxes[rng.randrange(len(boxes))]
    free = [p for p in _free_floor(level)
            if p not in level.box_start and p != level.player_start]
    if not free:
        return level
    return Level(level.walls, level.goals, level.player_start,
                 frozenset((level.box_start - {old}) | {rng.choice(free)}))


def _move_random_goal(level: Level, rng: random.Random) -> Level:
    goals = list(level.goals)
    if not goals:
        return level
    old = goals[rng.randrange(len(goals))]
    free = [p for p in _free_floor(level) if p not in level.goals]
    if not free:
        return level
    return Level(level.walls,
                 frozenset((level.goals - {old}) | {rng.choice(free)}),
                 level.player_start, level.box_start)


def _move_player(level: Level, rng: random.Random) -> Level:
    free = [p for p in _free_floor(level) if p not in level.box_start]
    if not free:
        return level
    return Level(level.walls, level.goals, rng.choice(free), level.box_start)


def _add_box_goal_pair(level: Level, rng: random.Random) -> Level:
    free_box = [p for p in _free_floor(level)
                if p not in level.box_start and p != level.player_start]
    free_goal = [p for p in _free_floor(level) if p not in level.goals]
    if not free_box or not free_goal:
        return level
    return Level(level.walls,
                 frozenset(level.goals | {rng.choice(free_goal)}),
                 level.player_start,
                 frozenset(level.box_start | {rng.choice(free_box)}))


def _remove_box_goal_pair(level: Level, rng: random.Random) -> Level:
    if len(level.box_start) <= 1:
        return level
    box = rng.choice(list(level.box_start))
    goal = rng.choice(list(level.goals))
    return Level(level.walls,
                 frozenset(level.goals - {goal}),
                 level.player_start,
                 frozenset(level.box_start - {box}))
