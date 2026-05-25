"""PlatformerDomain adapter for qd_core."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

from qd_core import Descriptor
import platformer_descriptors as d
from platformer_game import PlatLevel
from platformer_generator import random_level, mutate


@dataclass
class PlatformerDomain:
    H: int = 10
    W: int = 24
    node_budget: int = 4000

    def init_genome(self, rng: random.Random) -> PlatLevel:
        # Vary the init parameters a bit so the initial population spans
        # easy <-> challenging starting points.
        mean_h = rng.uniform(1.5, 3.5)
        plat_p = rng.uniform(0.02, 0.10)
        spike_p = rng.uniform(0.0, 0.08)
        pit_p = rng.uniform(0.05, 0.25)
        return random_level(self.H, self.W, rng,
                            mean_ground_height=mean_h,
                            platform_prob=plat_p,
                            spike_prob=spike_p,
                            pit_prob=pit_p)

    def mutate(self, parent: PlatLevel, rng: random.Random) -> PlatLevel:
        return mutate(parent, rng)

    def evaluate(self, level: PlatLevel):
        return d.evaluate(level, node_budget=self.node_budget)

    def render_genome(self, level: PlatLevel) -> str:
        return level.render()


DESCRIPTORS: Dict[str, Descriptor] = {
    "STR":    Descriptor("STR",    d.STR_SHAPE,    d.str_cell,    ("# pits", "height variance")),
    "PLAY":   Descriptor("PLAY",   d.PLAY_SHAPE,   d.play_cell,   ("plan length", "# jumps")),
    "SKILL":  Descriptor("SKILL",  d.SKILL_SHAPE,  d.skill_cell,  ("plan length", "log10(A* nodes)")),
    "COMMON": Descriptor("COMMON", d.COMMON_SHAPE, d.common_cell, ("# pits", "plan length")),
}
