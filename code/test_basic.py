"""Sanity tests for sokoban + solver."""
import time
from sokoban import Level
from solver import solve

TEST_TRIVIAL = """\
#######
#     #
# $.  #
#  @  #
#     #
#######
"""

TEST_TWO_BOXES = """\
#######
#  .  #
# $   #
#  @  #
#   $ #
#  .  #
#######
"""

TEST_UNSOLVABLE = """\
#####
#$  #
#@. #
#####
"""

TEST_CORNER_DEADLOCK = """\
######
#$   #
# @ .#
#    #
######
"""


def show(name, text):
    print(f"\n=== {name} ===")
    lvl = Level.parse(text)
    print(lvl.render())
    print(f"well_formed: {lvl.is_well_formed()}")
    print("deadlock map:")
    dead = lvl.deadlock_cells()
    for r in range(lvl.H):
        row = []
        for c in range(lvl.W):
            if lvl.walls[r, c]:
                row.append("#")
            elif dead[r, c]:
                row.append("x")
            else:
                row.append(".")
        print("  " + "".join(row))
    t0 = time.time()
    res = solve(lvl, node_budget=10_000)
    dt = time.time() - t0
    print(f"solve: solved={res.solved} plan_len={res.plan_len} pushes={res.n_pushes} nodes={res.nodes_expanded} timeout={res.timed_out} in {dt*1000:.1f}ms")


if __name__ == "__main__":
    for nm, t in [
        ("TRIVIAL", TEST_TRIVIAL),
        ("TWO_BOXES", TEST_TWO_BOXES),
        ("UNSOLVABLE", TEST_UNSOLVABLE),
        ("CORNER_DEADLOCK", TEST_CORNER_DEADLOCK),
    ]:
        show(nm, t)
