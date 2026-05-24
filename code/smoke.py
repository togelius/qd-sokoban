"""Quick smoke test: tiny run with each condition."""
import time
from qd import run_map_elites, run_random_search, project_to_common
from descriptors import (
    STRUCT_DESC, PLAY_DESC, SKILL_DESC,
    STRUCT_SHAPE, PLAY_SHAPE, SKILL_SHAPE,
)

H, W = 7, 7

print("=== STRUCT (200 gens) ===")
t0 = time.time()
a, log = run_map_elites(
    descriptor_fn=STRUCT_DESC["to_cell"],
    archive_shape=STRUCT_SHAPE, archive_name="STR",
    n_generations=200, n_init=40,
    H=H, W=W, seed=0, node_budget=2000, log_every=50,
)
print(f"  coverage={a.coverage()}/{STRUCT_SHAPE[0]*STRUCT_SHAPE[1]} qd={a.qd_score():.2f} max={a.max_fit():.2f} solver_calls={log.n_solver_calls[-1]} time={time.time()-t0:.1f}s")
print(f"  common-projected coverage={project_to_common(a).coverage()}/40")

print("\n=== PLAY (200 gens) ===")
t0 = time.time()
a, log = run_map_elites(
    descriptor_fn=PLAY_DESC["to_cell"],
    archive_shape=PLAY_SHAPE, archive_name="PLAY",
    n_generations=200, n_init=40,
    H=H, W=W, seed=0, node_budget=2000, log_every=50,
)
print(f"  coverage={a.coverage()}/{PLAY_SHAPE[0]*PLAY_SHAPE[1]} qd={a.qd_score():.2f} max={a.max_fit():.2f} solver_calls={log.n_solver_calls[-1]} time={time.time()-t0:.1f}s")
print(f"  common-projected coverage={project_to_common(a).coverage()}/40")

print("\n=== SKILL (200 gens) ===")
t0 = time.time()
a, log = run_map_elites(
    descriptor_fn=SKILL_DESC["to_cell"],
    archive_shape=SKILL_SHAPE, archive_name="SKILL",
    n_generations=200, n_init=40,
    H=H, W=W, seed=0, node_budget=2000, log_every=50,
)
print(f"  coverage={a.coverage()}/{SKILL_SHAPE[0]*SKILL_SHAPE[1]} qd={a.qd_score():.2f} max={a.max_fit():.2f} solver_calls={log.n_solver_calls[-1]} time={time.time()-t0:.1f}s")
print(f"  common-projected coverage={project_to_common(a).coverage()}/40")

print("\n=== RAND (200 evals) ===")
t0 = time.time()
a, log = run_random_search(
    n_generations=200, n_init=40,
    H=H, W=W, seed=0, node_budget=2000, log_every=50,
)
print(f"  coverage={a.coverage()}/40 qd={a.qd_score():.2f} max={a.max_fit():.2f} solver_calls={log.n_solver_calls[-1]} time={time.time()-t0:.1f}s")
