"""SokobanDomain — adapter that gives qd_core the protocol it needs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

from qd_core import Descriptor
import sokoban_descriptors as d
from sokoban_game import Level
from sokoban_generator import random_level, mutate


@dataclass
class SokobanDomain:
    H: int = 8
    W: int = 8
    node_budget: int = 8000
    init_box_choices: tuple = (1, 2, 3)
    init_wall_prob_range: tuple = (0.0, 0.3)

    def init_genome(self, rng: random.Random) -> Level:
        nb = rng.choice(self.init_box_choices)
        wp = rng.uniform(*self.init_wall_prob_range)
        return random_level(self.H, self.W, nb, wp, rng)

    def mutate(self, parent: Level, rng: random.Random) -> Level:
        return mutate(parent, rng)

    def evaluate(self, level: Level):
        return d.evaluate(level, node_budget=self.node_budget)

    def render_genome(self, level: Level) -> str:
        return level.render()


DESCRIPTORS: Dict[str, Descriptor] = {
    "STR":    Descriptor("STR",    d.STR_SHAPE,    d.str_cell,    ("# boxes", "wall density")),
    "PLAY":   Descriptor("PLAY",   d.PLAY_SHAPE,   d.play_cell,   ("plan length", "# pushes")),
    "SKILL":  Descriptor("SKILL",  d.SKILL_SHAPE,  d.skill_cell,  ("plan length", "log10(A* nodes)")),
    "COMMON": Descriptor("COMMON", d.COMMON_SHAPE, d.common_cell, ("# boxes", "plan length")),
}
