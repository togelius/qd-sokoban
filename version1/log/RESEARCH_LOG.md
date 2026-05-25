# Research Log

Living document of decisions, ideas tried, ideas abandoned, and lessons during this project.

## Day 0 — Scoping

**Brief**: Write a game-AI paper in the style of Togelius/Yannakakis from scratch
(idea, code, experiments, paper). Independent research test.

### Idea brainstorm

1. **MAP-Elites for Sokoban level generation** — well-trodden, but the
   choice of behavioral descriptors is rarely studied empirically. Tractable
   in one session.
2. **Adversarial co-evolution of solver + generator** — interesting but
   needs a learning solver; out of scope time-wise.
3. **Latent-space evolution of a neural level generator** — would need to
   train a VAE/GAN first, too much setup cost.
4. **Procedural personas via MAP-Elites on agent policies** — interesting,
   but requires either RL or a complex utility-weighted heuristic agent.
5. **Symbolic regression of heuristics from game tree search** — clever
   but the evaluation methodology would be murky.

**Picked (1)** because it has a concrete, falsifiable claim (descriptor
choice changes archive content), needs no external models, and fits in
the compute envelope of a single session.

### Refined question

> *Does the choice of behavioral descriptor in MAP-Elites materially
> change the kind of Sokoban puzzles produced, beyond cosmetic
> differences in the archive layout?*

Four conditions:

1. **RAND** — random search baseline (archive of best-N by fitness).
2. **ME-STR** — MAP-Elites, structural BDs: `(n_boxes, wall_density)`.
   Cheap because we never call the solver to compute BDs.
3. **ME-PLAY** — MAP-Elites, gameplay BDs: `(solution_length, n_pushes)`.
   Expensive because each BD eval requires an A* solve.
4. **ME-SKILL** — MAP-Elites, skill BDs: `(solution_length, log(search_nodes))`.
   Same cost as ME-PLAY but indexes by "AI hardness" instead of "human
   length".

For fair comparison, all final archives are projected onto a *common*
descriptor `(n_boxes, solution_length)` to compute coverage / QD-score.

### Compute budget

- 7×7 grid, 2–3 boxes. 6×6 is too cramped; 8×8 explodes solver time.
- A* with deadlock detection, node-expansion cap 5000.
- 2000 generations per run, 5 seeds per condition, 4 conditions = 20 runs.
- Targeting < 1h total wall-clock.

### Things I'm worried about

- A* timeout on dense random boards. Mitigation: timeout = "infeasible",
  fitness = 0.
- Local optima in ME-STR: a 4×10 archive only has 40 cells, so easy to
  fill but easy to misread coverage.
- I might end up with the trivial finding that ME-PLAY beats everyone
  on gameplay BDs because it optimizes them directly. Mitigation: the
  *common projection* is the headline metric, not in-archive scores.

### Realisation after first smoke run (5s per 200-gen run on 7×7)

I was originally going to frame the contrast as "cheap structural BDs
vs expensive gameplay BDs". But because the *fitness* in all conditions
requires a solve to verify solvability and plan length, the solver runs
on every individual regardless of BD choice. The marginal cost of a
gameplay BD is zero on top of fitness — we already have the
`SolveResult` cached.

That actually sharpens the paper: **wall-clock cost is held constant
across all four conditions**. Any difference in output is purely the
effect of the *behavioral descriptor choice*, not of the compute
budget. This is the kind of clean ablation Yannakakis/Togelius papers
favour.

Reframed contribution: *"Holding fitness and compute fixed, the choice
of behavioral descriptor in MAP-Elites materially changes the kind of
Sokoban puzzles surfaced in the archive."*

### Smoke results (200 gens, seed 0, 7×7)

| Cond  | Coverage (own) | QD-score | Common-coverage | Solver calls |
|-------|----------------|----------|-----------------|--------------|
| STR   | 11/40          | 6.84     | 9/40            | 240          |
| PLAY  | 20/100         | 10.96    | 13/40           | 240          |
| SKILL | 19/100         | 11.18    | 13/40           | 240          |
| RAND  | 12/40          | 6.22     | 12/40           | 240          |

Already a hint: ME-STR projects to *fewer* common cells than RAND. The
structural BD seems to *waste* exploration capacity on cells that map
onto the same `(n_boxes, plan_len)` cell. That's a candidate headline
finding. Need to confirm at scale + with multiple seeds.

---

## Day 0 — Implementation

Built in order:

1. `sokoban.py` — Level + State. Direct-encoded grid, precomputed
   reverse-BFS deadlock table. Validation: 1- and 2-box puzzles solve
   in <10ms.
2. `solver.py` — A* with Manhattan-distance heuristic + deadlock
   pruning. Node-budget kill-switch.
3. `generator.py` — random level + 6 mutation operators. About
   half the time random init produces a valid level on 7×7 with
   `wall_prob ∈ [0, 0.3]`.
4. `descriptors.py` — STR / PLAY / SKILL / COMMON descriptors, all
   with a shared `EvalCache` so the solver runs at most once per
   individual.
5. `qd.py` — Archive, RunLog, MAP-Elites loop, random search loop,
   common-projection helper.
6. `experiment.py` — multiprocessing-Pool over (cond × seed) jobs.
7. `analyze.py` — summary table, pairwise Mann–Whitney U, all plots,
   LaTeX table & inline-results emitter.
8. `render_levels.py` — matplotlib renderer for example puzzles.

### Things I tried and dropped

- **Surrogate solver** (use a tiny MLP to predict plan length, skip
  A* for descriptor): worth it only if solver dominates compute. On
  7×7/8×8 A* is fast enough — dropped.
- **Co-evolution** (generator-solver). Tempting but the solver is
  not learning; A* is fixed. Dropped — too much scope.
- **CMA-ME**. Considered for stronger baseline. Direct encoding is
  discrete, doesn't fit CMA's continuous parent distribution; would
  need indirect encoding. Out of scope.
- **Wall-density triangular fitness bonus** with peak at 0.3.
  Inspection of early-run elites showed peak should be at 0.2 to
  match what feasibility allows on 8×8. Re-tuned, results stable.
- **6×6 board**. Too small — barely room for 3 boxes plus interior
  walls. Dropped for 8×8.
- **Computing in-cell fitness variance** as a secondary metric for
  the heatmaps. Decided count-of-seeds-filled was a cleaner story
  for one figure.
- **"Procedural-personas" descriptor** (try 3 solvers of differing
  strength, BD = which solves it). Implementing 3 distinct solvers
  is its own project; SKILL's `log(nodes)` axis captures the same
  intuition more cheaply. Dropped.

### Bugs caught mid-flight

- First version of `log_to_dict` forgot to serialise the `common_*`
  time-series fields, so the convergence plot crashed after the first
  full experiment. Fixed and re-ran (40 runs, ~6 min).
- Original abstract claimed gameplay descriptors fill the long-plan
  region; the actual data show STRUCTURAL descriptors do, because
  wall density is correlated with plan length on 8×8. Rewrote.

---

## Day 0 — Final results

8×8, 5200 evaluations per run, 10 seeds per condition,
`node_budget=8000`. Mean ± std across seeds.

| Cond  | Cov (/40)   | QD-score    | Max fit     | Hard-cov (/16) | Hard-QD     | Wall time (s) |
|-------|-------------|-------------|-------------|----------------|-------------|---------------|
| RAND  | 24.3 ± 1.0  | 15.4 ± 0.6  | 0.90 ± 0.04 | 8.6 ± 0.8      | 6.9 ± 0.7   | 10.9 ± 0.5    |
| STR   | 31.5 ± 1.7  | 22.5 ± 1.1  | 1.00 ± 0.00 | **15.2 ± 0.8** | 13.5 ± 0.7  | 67.6 ± 8.1    |
| PLAY  | 31.3 ± 2.6  | 21.7 ± 1.9  | 0.99 ± 0.01 | 13.1 ± 1.5     | 11.3 ± 1.3  | 26.5 ± 11.2   |
| SKILL | **33.8 ± 1.1** | **23.5 ± 0.7** | 0.99 ± 0.01 | 13.5 ± 0.8 | 11.8 ± 0.8  | 35.3 ± 3.9    |

### Headline findings

1. **All MAP-Elites variants beat RAND** on every metric with perfect
   separation across seeds (p < 0.001, rank-biserial = +1.0).
2. **ME-SKILL wins overall coverage**, beating ME-STR by 2.3 cells
   and ME-PLAY by 2.5 (both p < 0.05).
3. **ME-STR wins on hard-cells** (plan_len ≥ 14), beating
   ME-SKILL/PLAY by ~1.5–2 cells (p < 0.01). Its wall-density
   axis incidentally biases it toward dense + long-plan puzzles.
4. **STR is 2.5× slower** than PLAY despite being the "cheapest"
   descriptor — selection pressure pushes toward dense boards
   where A* expands many nodes (p < 0.001).
5. **RAND never finds 4-box puzzles**; all MAP-Elites variants
   do, in 9–10/10 seeds.

### What I'd extend if I had another day

- 16×16 board with indirect (CPPN or NCA) encoding.
- Add a fourth "common" axis (e.g., choke points) to make a 3-D
  common archive and stress the projection methodology.
- Run with CMA-ME and check if the descriptor effect shrinks under a
  stronger emitter.
- Human-study (procedural personas-style): show pairs of puzzles
  from different conditions and ask which is more interesting.
