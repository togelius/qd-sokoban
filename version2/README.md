# qd-sokoban version 2

Extension of the [v1 paper](../version1/paper/paper.pdf) on behavioural
descriptor choice in MAP-Elites. v2 adds a second PCG domain
(tile-based platformer), doubles the seeds, uses proper effect sizes
and multiple-comparison corrections, and provides a mechanism analysis
showing that the descriptor's wall-clock cost in Sokoban is driven by
inflated solver-timeout rates, not by per-board complexity.

## Layout

```
version2/
├── code/
│   ├── qd_core.py            Domain-agnostic MAP-Elites + RAND
│   ├── stats_utils.py        Mann-Whitney + Cliff's δ + Holm-Bonferroni + bootstrap
│   ├── sokoban_*.py          Sokoban domain (game/solver/generator/descriptors/domain)
│   ├── platformer_*.py       Platformer domain
│   ├── run_experiment.py     Experiment driver (multiprocessing)
│   ├── analyze.py            Per-domain stats
│   ├── correlation_analysis.py   Descriptor-axis × log_nodes Spearman
│   ├── failure_analysis.py   Per-condition solvability / timeout breakdown
│   ├── make_figures.py       PDF figures
│   ├── make_tables.py        Auto-generated LaTeX tables
│   ├── render_examples.py    Top-fitness elite grid figure
│   ├── smoke_sokoban.py      Quick sanity check
│   └── smoke_platformer.py
├── results/
│   ├── sokoban/
│   │   ├── raw_runs.pkl              All archives + per-evaluation records
│   │   ├── summary.json              Per-run summaries
│   │   ├── stats_summary.txt         Per-metric mean/SD/CI table
│   │   ├── stats_pairwise.txt        Pairwise Mann-Whitney with Cliff's δ
│   │   ├── mechanism.json            Per-condition retention bias
│   │   ├── failure_breakdown.json    Solvability / timeout rates
│   │   ├── descriptor_hardness_correlation.json
│   │   └── figures/                  PDF figures
│   ├── platformer/                   (same shape)
│   └── cross_domain_summary.pdf
├── paper/
│   ├── paper.tex                     The paper
│   ├── references.bib                Bibliography
│   ├── results_table.tex             Auto-generated summary table
│   ├── results_pairwise.tex          Auto-generated pairwise table
│   └── results_inline.tex            Auto-generated \newcommand macros
└── log/
    └── RESEARCH_LOG.md               Decisions, dead ends, raw findings
```

## Reproducing

```bash
cd code
# Sokoban: ~11 min on 8 cores
python3 run_experiment.py --domain sokoban --processes 8

# Platformer: ~2 min on 4 cores
python3 run_experiment.py --domain platformer --processes 4

# Stats + figures
python3 analyze.py --domain sokoban
python3 analyze.py --domain platformer
python3 failure_analysis.py --domain sokoban
python3 failure_analysis.py --domain platformer
python3 correlation_analysis.py --domain sokoban
python3 correlation_analysis.py --domain platformer
python3 make_figures.py --domain sokoban
python3 make_figures.py --domain platformer
python3 make_figures.py --combined
python3 render_examples.py --domain sokoban
python3 render_examples.py --domain platformer
python3 make_tables.py

# Paper
cd ../paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Add `--quick` to `run_experiment.py` for a 3-seed sanity check
(~30 s for either domain).

## Key results in one number

Sokoban's structural descriptor (\#boxes, wall density) runs
**2.2× slower** in wall-clock than the gameplay descriptor
(plan length, push count), even though it never invokes the
solver to compute its own coordinates. The mechanism: it pushes
mutations toward dense-wall elites, mutations from which produce
a **20.0% infeasibility rate** versus 12.9% for the gameplay
descriptor — and every infeasible attempt consumes the full
8000-node solver budget.

The same descriptor causes essentially no wall-clock overhead in
the platformer (within 1.4× of the fastest), because the
platformer's solver is uniformly fast and infeasibility rates are
uniform across conditions (5-6%).

## Dependencies

`numpy`, `scipy`, `matplotlib`, plus `pdflatex` (with
`texlive-publishers` for IEEEtran) for the paper.
