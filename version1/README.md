# Behavioral Descriptor Choice in MAP-Elites for Sokoban Puzzle Generation

A small, self-contained research project comparing three behavioral
descriptor families for MAP-Elites-driven Sokoban level generation,
plus a random-search baseline. Holds fitness, encoding, and compute
budget fixed; varies only the descriptor.

The accompanying paper is in `paper/paper.pdf`.

## Layout

```
paper_qd_sokoban/
├── code/                      # all source
│   ├── sokoban.py             # Level, State, mechanics, deadlock map
│   ├── solver.py              # A* with deadlock-pruned heuristic
│   ├── generator.py           # random init + 6 mutation operators
│   ├── descriptors.py         # STR / PLAY / SKILL / COMMON descriptors + fitness
│   ├── qd.py                  # MAP-Elites and random-search loops + archive class
│   ├── experiment.py          # multiprocessing experiment driver
│   ├── analyze.py             # tables, stats, plots, LaTeX emitters
│   ├── render_levels.py       # matplotlib renderer for example puzzles
│   ├── test_basic.py          # smoke tests for sokoban/solver
│   └── smoke.py               # tiny smoke run of all 4 conditions
├── results/                   # all generated outputs (gitignored except summaries)
│   ├── raw_results.pkl        # serialised archives + logs for all 40 runs
│   ├── summary.json           # per-run scalar metrics
│   ├── summary_table.txt      # human-readable table
│   ├── stats.txt              # pairwise Mann-Whitney U
│   ├── *.pdf                  # figures (heatmaps, convergence, bars, time)
│   ├── examples/              # ASCII renderings of top puzzles per condition
│   └── figures/               # rendered example-puzzle PDFs
├── paper/                     # IEEE conference paper sources + compiled PDF
│   ├── paper.tex
│   ├── results_table.tex      # auto-generated
│   ├── results_inline.tex     # auto-generated
│   ├── references.bib
│   └── paper.pdf
└── log/
    └── RESEARCH_LOG.md        # ideas, decisions, dead ends
```

## Reproducing

```bash
cd code
python3 experiment.py        # ~6 min on 4 cores
python3 analyze.py           # generates plots and tables
python3 render_levels.py     # generates example-puzzle figures
cd ../paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

`experiment.py --quick` runs a sanity-check version (3 seeds, 7×7,
1000 evaluations) in ~10 s.

## Key result, in one number

The "cheap" structural descriptor is **2.5× slower** in wall-clock
than the gameplay descriptor, despite needing no extra solver calls
per evaluation. It selects for the boards that A* finds hardest to
solve.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pdflatex` (with `texlive-publishers`
for IEEEtran).
