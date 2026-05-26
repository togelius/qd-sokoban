"""Cross-domain consistency: Stouffer combined p-values for key pairwise
contrasts whose direction is hypothesised to hold across both domains.

For each hypothesis (e.g. "COMMON > PLAY on common_coverage"), we
combine the Sokoban and platformer p-values via Stouffer's Z-method
(equal weights). The combined p is one-sided in the hypothesised
direction. We require the per-domain effect signs to agree before
combining; if they disagree the combined result is reported as
inconclusive.

Outputs ../results/cross_domain_consistency.json (and prints a table).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from analyze import metric_table, load_runs, pairwise_table
from stats_utils import stouffer_combine


HYPOTHESES = [
    # (name, metric, A, B)  — H1: A > B
    ("COMMON > RAND on coverage",          "common_coverage", "COMMON", "RAND"),
    ("COMMON > RAND on QD",                "common_qd_score", "COMMON", "RAND"),
    ("COMMON > PLAY on coverage",          "common_coverage", "COMMON", "PLAY"),
    ("COMMON > PLAY on QD",                "common_qd_score", "COMMON", "PLAY"),
    ("COMMON > STR  on coverage",          "common_coverage", "COMMON", "STR"),
    ("COMMON > SKILL on coverage",         "common_coverage", "COMMON", "SKILL"),
    ("PLAY < SKILL on coverage",           "common_coverage", "SKILL",  "PLAY"),
    ("PLAY < STR   on coverage",           "common_coverage", "STR",    "PLAY"),
    ("PLAY < COMMON on QD",                "common_qd_score", "COMMON", "PLAY"),
    ("STR retains harder boards than PLAY",
                                            "mech_logn",       "STR",    "PLAY"),
]


def signed_p(table, metric, a, b):
    """One-sided p for H1: mean(a) > mean(b). Two-sided / 2 if sign agrees."""
    rows = pairwise_table(table[metric])
    pair = tuple(sorted([a, b]))
    r = next((x for x in rows if (x.a, x.b) == pair), None)
    if r is None:
        return None, None, None
    # canonical (r.a, r.b). delta is mean(r.a) - mean(r.b).
    delta_signed = r.cliffs_delta if (r.a, r.b) == (a, b) else -r.cliffs_delta
    # one-sided p in direction (a > b)
    if delta_signed > 0:
        p_one = r.p_value / 2.0
    elif delta_signed < 0:
        p_one = 1.0 - r.p_value / 2.0
    else:
        p_one = 0.5
    return p_one, delta_signed, r.p_value


def mech_logn_table(runs):
    """Build a metric_table-like dict for the per-seed mean log_nodes of
    retained elites. Used for the 'STR retains harder boards' hypothesis."""
    by_cond = defaultdict(list)
    for r in runs:
        cells = r["common_archive"]["cells"]
        logs = [np.log10(c["nodes_expanded"])
                for c in cells if c["solvable"] and c["nodes_expanded"] > 0]
        if logs:
            by_cond[r["cond"]].append(float(np.mean(logs)))
    return {"mech_logn": dict(by_cond)}


def main():
    soko = load_runs("../results/sokoban")
    plat = load_runs("../results/platformer")
    soko_table = {**metric_table(soko), **mech_logn_table(soko)}
    plat_table = {**metric_table(plat), **mech_logn_table(plat)}

    print(f"{'hypothesis':<42}  {'soko p':>9} {'soko δ':>8}  "
          f"{'plat p':>9} {'plat δ':>8}  {'combined p':>11}  consistent?")
    out = []
    for name, metric, a, b in HYPOTHESES:
        p_s, d_s, _ = signed_p(soko_table, metric, a, b)
        p_p, d_p, _ = signed_p(plat_table, metric, a, b)
        consistent = (d_s is not None and d_p is not None
                      and np.sign(d_s) == np.sign(d_p) and d_s > 0)
        combined = stouffer_combine([p_s, p_p]) if consistent else None
        marker = "yes" if consistent else "no (sign mismatch)"
        cs = "—" if combined is None else f"{combined:.2g}"
        print(f"{name:<42}  {p_s:>9.2g} {d_s:>+8.2f}  "
              f"{p_p:>9.2g} {d_p:>+8.2f}  {cs:>11}  {marker}")
        out.append(dict(hypothesis=name, metric=metric, a=a, b=b,
                        sokoban=dict(p=float(p_s), delta=float(d_s)),
                        platformer=dict(p=float(p_p), delta=float(d_p)),
                        combined_p=(float(combined) if combined is not None else None),
                        consistent=bool(consistent)))
    with open("../results/cross_domain_consistency.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote ../results/cross_domain_consistency.json")


if __name__ == "__main__":
    main()
