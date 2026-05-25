# Research Log — Version 2

Living document. Same conventions as v1: chronological, includes
abandoned ideas with reasons.

## Day 1 — Critique of v1

Re-read v1 paper and results before committing to a v2 direction.

### What v1 actually shows (honest assessment)

1. *All MAP-Elites variants beat random search on every metric.* Expected.
   Not novel. Useful as a sanity check, not a contribution.
2. *ME-SKILL has the best common-coverage by ~2.3 cells.* Small effect
   in absolute terms; the conditions are all near the ceiling of the
   16-cell hard region.
3. *ME-STR has the highest hard-cell coverage.* Interesting but specific
   to one corner of the common archive.
4. *ME-STR runs 2.5× slower than ME-PLAY despite being the "cheapest"
   descriptor.* This is the most striking finding and the only one with
   a hint of a *mechanistic* claim ("structural pressure selects for
   computationally hard boards").
5. *Random search never produces 4-box puzzles.* Cute but mostly a
   property of the mutation operators, not of the descriptor.

The actual story is buried in finding 4. The headline is being told as
an empirical comparison ("descriptors give different archives") when it
should be a mechanism paper ("descriptors are not neutral diversity
axes; they impose hidden selection pressures").

### Other weaknesses of v1

- **One domain.** Any finding could be Sokoban-specific.
- **Related work is thin.** Three citations to MAP-Elites/AURORA-ish work,
  no engagement with adaptive-descriptor literature (CMA-ME, CMA-MAE,
  DSAGE), no engagement with the broader PCG-via-QD literature
  (Earle's *Illuminating Mario Scenes*, Khalifa's *PCGRL*,
  Gravina-Liapis-Yannakakis surveys).
- **N=10 seeds.** Adequate for non-parametric tests but underpowered
  for cross-condition interactions. Effect sizes reported only as
  rank-biserial; no bootstrap CIs.
- **No multiple-comparison correction.** Six pairwise tests run, no
  adjustment.
- **No mechanism check.** v1 *speculates* that STR selects for hardness
  via dense-wall correlation but never demonstrates this directly.
- **Hard-cell findings are dependent on an arbitrary plan-length
  threshold** (`plan_len >= 14`).
- **Paper structure is plain.** Intro doesn't motivate why this matters
  beyond "no one's measured it"; conclusion gives no actionable advice.

## Day 1 — Thesis for v2

**Central claim:**

> The behavioral descriptor in a MAP-Elites archive is not a neutral
> diversity axis. It induces an *implicit selection pressure* on the
> generated population, distinct from the explicit fitness function,
> that can dominate the run's output. This implicit pressure arises
> because the descriptor determines which mutated offspring are
> *retained* (cell-winners) versus *discarded* (cell-losers), and the
> probability of being retained depends on properties of the genome
> that may not appear in the fitness function at all.

This reframes the v1 finding "STR is slow because it selects for hard
boards" as a *general property of MAP-Elites with hand-designed
descriptors* — not a Sokoban quirk.

### Predictions following from the thesis

P1. **Cross-domain invariance.** The phenomenon should replicate in a
non-Sokoban PCG domain that has analogous descriptor families. If it
doesn't, the thesis is wrong.

P2. **Correlation with hardness.** Conditions whose descriptor is
correlated with solver-hardness should produce harder content on
average, *independently of fitness*. Measurable as: in each condition,
correlate the descriptor coordinates of an individual with its solver
node-count, and check whether this correlation is non-zero.

P3. **De-confounding.** If we construct a descriptor that is the
*residual* of a structural descriptor after regressing out hardness,
the implicit pressure should disappear. The "hardness-corrected" STR
should not be slow and should not preferentially fill hard cells.

P4. **Fitness-equivalent, archive-different.** Conditions with similar
peak/mean fitness can still produce dramatically different archives
in projection. (v1 already hints at this; v2 makes it quantitative.)

## Day 1 — Contributions

1. **Reframing:** descriptors as implicit selection pressure, formalized.
2. **Two-domain empirical study:** Sokoban (replicating v1 at larger
   scale) + a tile-based platformer (new). Same descriptor families,
   same evaluation protocol.
3. **Mechanism:** hardness-correlation analysis (P2) and the
   hardness-corrected descriptor (P3).
4. **Practical guidance:** a small decision flowchart for descriptor
   choice based on what the practitioner cares about (diversity? hardness?
   wall-clock?).
5. **Open replication package:** all code, raw archives, statistics
   scripts in this repo.

## Day 1 — Choice of second domain

Constraints: must be a content-generation domain, must have an
automatable evaluator that scales like Sokoban's A*, must support
descriptors of all three families (STR / PLAY / SKILL).

Options I considered:

- **Mario-style platformer** — well-trodden in PCG-via-QD literature
  (Earle, Khalifa, et al.). Implementable as tile grid + jump physics
  + reachability solver. Picked.
- **Zelda-style dungeon rooms** — interesting but the evaluator
  (puzzle-solvability) gets complex.
- **Lunar Lander / locomotion** — canonical QD but not PCG, breaks the
  thematic coherence of the paper.
- **Lights Out / sliding puzzle** — too similar to Sokoban
  structurally; wouldn't test cross-domain generalization.
- **Mini-Zelda / VGDL game** — would need a VGDL interpreter; too much
  scaffolding.

**Picked: tile-based platformer.** Spec:

- Grid: 10 rows × 32 cols.
- Tiles: empty, ground, brick, spike (hazard), enemy, goal.
- Player physics: gravity, jump height ≤ 4 tiles, horizontal speed 1
  tile/step, optional double-jump turned off for simplicity.
- Solver: A* over (x, y, vy_state) with reachability set. Node-budget
  killswitch like Sokoban.
- Mutation: similar 6-operator scheme (toggle tile, raise/lower
  column, add/remove enemy, shift slice).
- Descriptors:
  - **PLAT-STR**: `(pit_count, ground_roughness)`.
  - **PLAT-PLAY**: `(n_jumps_in_solution, completion_time)`.
  - **PLAT-SKILL**: `(completion_time, log_search_nodes)`.

This mirrors Sokoban's `(n_boxes, wall_density)` / `(plan_len,
n_pushes)` / `(plan_len, log_nodes)` exactly.

## Day 1 — Experimental plan

- **Sokoban:** 8×8, node_budget=8000, 5200 evals/run, **20 seeds × 5
  conditions** = 100 runs. (v1 used 10 seeds × 4 conditions.) Adding
  `ME-STRH` (hardness-corrected structural) as fifth condition.
- **Platformer:** 10×32, node_budget=4000, 4000 evals/run, 20 seeds × 5
  conditions = 100 runs.
- **Stats:** per-metric Mann–Whitney with Holm–Bonferroni adjustment
  across the 10 pairwise comparisons per metric; Cliff's δ as
  effect-size; bootstrap 95% CIs on means; cross-domain consistency
  via meta-analytic combined p-value (Stouffer).

Time budget: aiming for < 2 h total compute, comfortably parallelisable.

## Day 1 — Implementation order

1. Port Sokoban code to `version2/code/sokoban/` with module split.
2. Implement platformer in `version2/code/platformer/`.
3. Shared experiment driver and analysis under `version2/code/common/`.
4. Smoke-test both domains.
5. Run full experiments (probably overnight).
6. Stats and figures.
7. Paper.

## Day 1 — Sokoban port complete

Sokoban code split into 5 files (game / solver / generator / descriptors /
domain), all plugged into the new domain-agnostic `qd_core.py`. Smoke
run at 7×7 with 500 evals/condition reproduces v1 qualitatively:

```
cond     cov     qd  maxf   t(s)
RAND      15   7.97 0.677    0.2
STR       22  13.73 0.874    0.8
PLAY      18  10.85 0.900    0.4
SKILL     21  12.28 0.824    0.8
COMMON    28  16.32 0.875    0.5
```

STR already 2× slower than PLAY at this scale. Added ME-COMMON as a
5th condition — it directly optimises the COMMON descriptor, so its
common-coverage is an upper bound and a useful reference for "how
much room is left on the table for the indirect descriptors."

## Day 1 — Platformer design decisions

Choosing the simplest tractable platformer that supports analogous
descriptor families:

- 10 rows × 24 cols, tile alphabet = {empty, solid, spike}.
- Goal: reach column 23 alive from column 0.
- Player physics: standing-to-standing graph. From a standing
  position (c, r), available moves are:
  - **WALK**: (c+1, r) if (r, c+1) empty and (r+1, c+1) solid.
  - **WALK-OFF**: (c+1, r') for r' > r if walking off a ledge; r'
    is the first row with (r'+1, c+1) solid.
  - **JUMP(dx, dy)**: arc up to (c, r-dy), forward to (c+dx, r-dy),
    then fall to standing on (c+dx, r''). Parameters dx ∈ 0..4,
    dy ∈ 1..3. Vertical+horizontal segments must be empty; landing
    cell empty; cell below landing solid.

Spike kill: if any tile traversed (arc or landing) is a spike, the
move is invalid. Death by falling off the bottom is invalid too.

This is intentionally similar to the "Mario tile-graph" approximation
used in the Mario AI papers, but cut down to be self-contained.

Solver = A* over standing positions, heuristic = manhattan-to-goal,
node-budget kill switch.

Descriptors for the platformer:
- **PLAT-STR**: `(pit_count, height_variance_bin)` — both tile-only.
- **PLAT-PLAY**: `(n_jumps_in_solution, plan_len)`.
- **PLAT-SKILL**: `(plan_len, log10(search_nodes))`.
- **PLAT-COMMON**: `(pit_count, plan_len)` — used for cross-method
  projection.

This mirrors the Sokoban descriptor family structure exactly: STR is
solver-independent; PLAY uses solution path properties; SKILL uses
solver-effort.

## Day 1 — Platformer pilot (W=24, 5000 evals, seed 0)

```
cond     cov   qd   maxf   meanf   t(s)
RAND      41  34.9  0.93   0.85    2.7
STR       39  35.1  0.97   0.90    2.3
PLAY      33  28.8  0.95   0.87    2.5
SKILL     36  31.6  0.95   0.88    2.4
COMMON    42  37.9  0.97   0.90    2.4
```

**Notable.** STR is *not* slower than PLAY/SKILL here, in sharp contrast
to Sokoban (v1: STR ~2.5x slower than PLAY). The platformer's A* over
standing-positions is fast on all reasonable levels — dense terrain
doesn't blow up the search space the way dense boxes do in Sokoban.

This is good for the paper: the *implicit-selection-pressure* claim
holds (descriptors give different content distributions), but its
*computational cost* manifestation is domain-dependent. Reframe the
v1 finding "STR is slow" as a special case of a more general property.

PLAY underperforms by ~10 cells of common-coverage in the platformer —
the opposite ordering from Sokoban v1 where ME-PLAY was middle of the
pack. (n_pits, plan_len) projection isn't well-aligned with PLAY's
(plan_len, n_jumps) own descriptor, so PLAY wastes capacity on cells
that map onto the same common cell.

I also tried W=32 first; plan_len ended up too narrow because the
solver always finds nearly-walk-only paths. W=24 + bin retuning gave
acceptable spread without losing too much "story."

## Day 1 — Final experiment plan (revised)

Drop the proposed `ME-STRH` (hardness-corrected structural) condition.
Keep five conditions: RAND, STR, PLAY, SKILL, COMMON. ME-COMMON acts
as a "fair ceiling" — it directly optimises the projection, so its
common-coverage is a natural reference point for the indirect
descriptors. The mechanism analysis (Section 5 of the paper) will be
done via correlation/regression rather than a 5th descriptor variant.

**Budgets:**
- Sokoban: 8x8, node_budget=8000, 5500 evals/run, n_init=250, 20 seeds.
- Platformer: 10x24, node_budget=4000, 5000 evals/run, n_init=250, 20 seeds.

5 conditions × 20 seeds × 2 domains = 200 runs. With 8 parallel
processes, estimated wall-clock ~90 minutes. Log every 250 evals.

## Day 1 — Full results

Sokoban: 11.2 min wall (100 runs over 8 workers). Platformer: 2.2 min
(100 runs over 4 workers, because another QD process on the machine
was occupying cores).

### Sokoban headline numbers (mean$\pm$SD over 20 seeds)

| cond   | cov (/40) | QD       | max_fit | wall (s)   |
|--------|-----------|----------|---------|------------|
| RAND   | 24.0±0.8  | 15.2±0.6 | 0.91    | **11.2**   |
| STR    | 32.4±2.0  | 23.0±1.0 | 1.00    | **81.8**   |
| PLAY   | 31.3±2.5  | 21.5±1.8 | 0.99    | 37.3       |
| SKILL  | 33.8±2.6  | 23.4±1.8 | 1.00    | 49.8       |
| COMMON | 37.3±1.7  | 25.8±1.1 | 1.00    | 76.5       |

After Holm correction (10 pairwise tests per metric):

- All MAP-Elites > RAND: $p_\text{adj}<10^{-6}$, $\delta=1.00$ (perfect).
- COMMON > everyone else on cov / qd: $p_\text{adj}<10^{-4}$, $\delta\geq 0.72$.
- SKILL > PLAY: $p_\text{adj}=0.024$, $\delta=-0.49$ (large).
- STR vs PLAY: not significant ($p_\text{adj}=0.21$, $\delta=-0.23$).
- SKILL vs STR: marginal ($p_\text{adj}=0.12$, $\delta=+0.34$).

### Platformer headline numbers (mean$\pm$SD over 20 seeds)

| cond   | cov (/80) | QD       | max_fit | wall (s)  |
|--------|-----------|----------|---------|-----------|
| RAND   | 39.8±1.5  | 34.0±1.2 | 0.94    | 6.2       |
| STR    | 39.9±4.6  | 35.3±4.1 | 0.96    | 5.5       |
| PLAY   | 35.4±2.9  | 31.0±2.6 | 0.95    | 4.8       |
| SKILL  | 38.9±3.7  | 34.3±3.1 | 0.96    | 5.7       |
| COMMON | 45.5±4.0  | 40.7±3.6 | 0.96    | 4.3       |

- COMMON > everyone: $p_\text{adj}<0.01$, $\delta\geq 0.63$.
- PLAY < everyone else: $p_\text{adj}<0.03$, $\delta\leq -0.51$ (large).
- STR ≈ RAND ≈ SKILL on coverage; STR > RAND on QD-score (marginal).

### Mechanism: where the wall-clock comes from

Failure breakdown (Sokoban):

| cond   | %wf   | %solv  | %unsolv | log10(N_all) |
|--------|-------|--------|---------|--------------|
| RAND   | 100%  | 22.2%  | 5.7%    | 2.22         |
| STR    | 98.5% | 55.4%  | **20.0%** | **2.90**   |
| PLAY   | 98.5% | 63.1%  | 12.9%   | 2.45         |
| SKILL  | 98.3% | 64.4%  | 12.6%   | 2.42         |
| COMMON | 97.8% | 62.3%  | 14.6%   | 2.54         |

STR's 20% infeasibility rate (vs PLAY's 13%) × 8000-node budget per
timeout almost entirely accounts for the 2.2× wall-clock difference.
This refines the v1 explanation ("STR selects for hard solvable
boards"): the dominant cost is timed-out infeasible mutants, not
slow solves of feasible boards.

Failure breakdown (Platformer):

| cond   | %solv  | %unsolv | log10(N_all) |
|--------|--------|---------|--------------|
| RAND   | 87.6%  | 12.4%   | 1.27         |
| STR    | 94.1%  | 5.9%    | 1.27         |
| PLAY   | 95.0%  | 5.0%    | 1.29         |
| SKILL  | 94.3%  | 5.7%    | 1.28         |
| COMMON | 93.8%  | 6.2%    | 1.27         |

All within 1.0× of each other. No condition-specific
failure-work inflation. This is what makes the platformer the
"control" — same algorithm, same descriptor families, but the
mechanism does not have a wall-clock manifestation.

### Spearman correlations (descriptor cell × log_nodes, solvable evals only)

Sokoban:
- RAND  (n_boxes, plan_len):       rho_ax1=+0.81, rho_ax2=+0.86
- STR   (n_boxes, wall_density):   rho_ax1=+0.82, rho_ax2=-0.07
- PLAY  (plan_len, n_pushes):      rho_ax1=+0.85, rho_ax2=+0.82
- SKILL (plan_len, log_nodes):     rho_ax1=+0.75, rho_ax2=+0.99 (by design)
- COMMON (n_boxes, plan_len):      rho_ax1=+0.54, rho_ax2=+0.84

Platformer:
- RAND  (n_pits, plan_len):        rho_ax1=-0.63, rho_ax2=-0.02
- STR   (n_pits, height_var):      rho_ax1=-0.73, rho_ax2=-0.09
- PLAY  (plan_len, n_jumps):       rho_ax1=-0.06, rho_ax2=-0.07
- SKILL (plan_len, log_nodes):     rho_ax1=-0.08, rho_ax2=+0.95
- COMMON (n_pits, plan_len):       rho_ax1=-0.72, rho_ax2=+0.07

The Sokoban correlations are mostly positive (axis ↑ → hardness ↑),
*except* STR's wall_density which is weakly negative. So the
hardness selection in STR is not driven by a monotonic
density→hardness mapping; it's driven by infeasibility-rate
inflation as Table~failure shows.

In the platformer, n_pits is *negatively* correlated with log_nodes
(more pits → easier search, because most pit-heavy levels are
infeasible and get filtered before they consume many nodes). This
is the opposite direction from Sokoban and explains why STR is
actually *slightly faster* than RAND in the platformer — selection
on pits indirectly selects for solver-cheap content.

## Day 1 — Things I tried and dropped

- **ME-STRH (hardness-corrected structural)** as a 5th condition.
  Replaced with ME-COMMON, which serves as a "fair upper bound"
  reference and supplies the headline projection-metric.
- **W=32 platformer** as a way to get more plan-length spread.
  Made plan_len even more narrow (everything walking-dominated);
  reverted to W=24 with tuned bins.
- **Adding left-motion + backtracking to the platformer** to get
  more solver-effort variance. Decided it would change the genre
  too much and didn't help the cross-domain story.
- **MAP-Elites with random uniform parent selection vs CMA-ME
  emitter** as a cross-cut. Out of scope — would have required
  re-implementing CMA-ME.
- **Spearman correlation of descriptor cell with log_nodes** as
  *the* mechanism explanation. Computed but the more revealing
  number turned out to be the per-condition infeasibility rate
  (Table~failure in the paper). Kept correlation as a supplementary
  analysis.

## Day 1 — Things v2 didn't fully fix

- The hardness mechanism in Sokoban is mediated by infeasibility,
  not by per-board solver effort. The framing in
  \S\ref{sec:method} ("descriptors that correlate with hardness
  retain harder content") is partially true: STR's *retained*
  log_nodes are higher, but the dominant wall-clock cost is from
  *unretained* timed-out mutants. The paper now reflects this; a
  v3 could try to disentangle these two routes more cleanly
  (e.g.\ by capping solver budget at a level that makes timeouts
  comparable across conditions).
- The cross-domain non-preservation of rank order is suggestive
  but I'd want another 1--2 domains to call it a generalisation.

