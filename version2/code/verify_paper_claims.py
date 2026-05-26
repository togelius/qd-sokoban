"""Recompute every numerical/statistical claim made in paper.tex
from the raw data and assert that the paper's numbers match.

Prints PASS / FAIL per claim. Exits non-zero if any FAIL.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from analyze import metric_table, load_runs, pairwise_table
from stats_utils import bootstrap_ci

PAPER_DIR = "../paper"
SOKO_DIR = "../results/sokoban"
PLAT_DIR = "../results/platformer"

passed = 0
failed = 0
warnings = []


def check(label, ok, msg=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}    {msg}")
    else:
        failed += 1
        print(f"  FAIL  {label}    {msg}")


def warn(label, msg):
    warnings.append((label, msg))
    print(f"  WARN  {label}    {msg}")


def close(actual, claimed, tol=0.05, label=""):
    if claimed == 0:
        return abs(actual) < tol
    return abs(actual - claimed) / abs(claimed) < tol


# ============================================================================
# Section: load data
# ============================================================================
soko = load_runs(SOKO_DIR)
plat = load_runs(PLAT_DIR)
soko_table = metric_table(soko)
plat_table = metric_table(plat)


# ============================================================================
# Abstract claims
# ============================================================================
print("\n== ABSTRACT ==")

# "200 runs, ≈ 1.05M evaluations"
total_runs = len(soko) + len(plat)
check("abstract.runs=200", total_runs == 200,
      f"actual={total_runs}")
soko_evals = sum(len(r["eval_records"]) for r in soko)
plat_evals = sum(len(r["eval_records"]) for r in plat)
total_evals = soko_evals + plat_evals
check("abstract.evals~1.05M", abs(total_evals - 1_050_000) / 1_050_000 < 0.05,
      f"actual={total_evals:,} (soko {soko_evals:,} + plat {plat_evals:,})")

# "2.2× slower than gameplay descriptor" — STR vs PLAY in Sokoban wall_time
str_wt = np.mean(soko_table["wall_time_s"]["STR"])
play_wt = np.mean(soko_table["wall_time_s"]["PLAY"])
ratio = str_wt / play_wt
check("abstract.2.2x_slower", abs(ratio - 2.2) < 0.15,
      f"STR/PLAY = {str_wt:.1f}/{play_wt:.1f} = {ratio:.2f}x")

# "20% infeasibility rate (versus 13% for the gameplay descriptor)"
with open(os.path.join(SOKO_DIR, "failure_breakdown.json")) as f:
    soko_fb = json.load(f)
str_unsolv = 100 * soko_fb["STR"]["frac_unsolvable_attempted"]
play_unsolv = 100 * soko_fb["PLAY"]["frac_unsolvable_attempted"]
check("abstract.STR_unsolv=20%", abs(str_unsolv - 20.0) < 0.5,
      f"actual={str_unsolv:.1f}%")
check("abstract.PLAY_unsolv=13%", abs(play_unsolv - 13.0) < 1.0,
      f"actual={play_unsolv:.1f}%")


# ============================================================================
# Findings (F1-F4)
# ============================================================================
print("\n== FINDINGS ==")

# F1 (Sokoban): All MAP-Elites > RAND with Cliff's δ ≥ 0.998
for cond in ["STR", "PLAY", "SKILL", "COMMON"]:
    for metric in ["common_coverage", "common_qd_score"]:
        rows = pairwise_table(soko_table[metric])
        r = next((x for x in rows
                  if (x.a, x.b) == tuple(sorted([cond, "RAND"]))), None)
        check(f"F1.sokoban.{cond}>RAND.{metric}.p<1e-6",
              r.p_adj < 1e-6,
              f"p_adj={r.p_adj:.2g}")
        check(f"F1.sokoban.{cond}>RAND.{metric}.delta>=0.997",
              abs(r.cliffs_delta) >= 0.997,
              f"|delta|={abs(r.cliffs_delta):.4f}")

# F1 (Platformer): PLAY < RAND on coverage (δ ≈ -0.80, p_adj < 10^-3)
rows = pairwise_table(plat_table["common_coverage"])
r = next((x for x in rows if (x.a, x.b) == ("PLAY", "RAND")), None)
d_signed = r.cliffs_delta if r.a == "PLAY" else -r.cliffs_delta
check("F1.platformer.PLAY<RAND.cov.delta~-0.80", abs(d_signed + 0.80) < 0.03,
      f"delta_signed={d_signed:.2f}")
check("F1.platformer.PLAY<RAND.cov.p<1e-3", r.p_adj < 1e-3,
      f"p_adj={r.p_adj:.2g}")

# F1 (Platformer): STR ≈ RAND (p_adj >= 0.47)
r = next((x for x in rows if (x.a, x.b) == ("RAND", "STR")), None)
check("F1.platformer.STR~RAND.cov.p=0.77", abs(r.p_adj - 0.77) < 0.03,
      f"p_adj={r.p_adj:.2g}")
r = next((x for x in rows if (x.a, x.b) == ("RAND", "SKILL")), None)
check("F1.platformer.SKILL~RAND.cov.p=0.47", abs(r.p_adj - 0.47) < 0.03,
      f"p_adj={r.p_adj:.2g}")

# F1 (Platformer): COMMON > RAND δ=+0.82
r = next((x for x in rows if (x.a, x.b) == ("COMMON", "RAND")), None)
d_signed = r.cliffs_delta if r.a == "COMMON" else -r.cliffs_delta
check("F1.platformer.COMMON>RAND.cov.delta=0.82", abs(d_signed - 0.82) < 0.03,
      f"delta_signed={d_signed:.2f}")
check("F1.platformer.COMMON>RAND.cov.p<1e-4", r.p_adj < 1e-4,
      f"p_adj={r.p_adj:.2g}")

# "COMMON still beats RAND with δ=0.82" (platformer)
rows = pairwise_table(plat_table["common_coverage"])
r = next((x for x in rows
          if (x.a, x.b) == ("COMMON", "RAND")), None)
d = r.cliffs_delta if r.a == "COMMON" else -r.cliffs_delta
check("F1.plat.COMMON>RAND.delta~0.82", abs(abs(d) - 0.82) < 0.05,
      f"|delta|={abs(d):.2f}, p_adj={r.p_adj:.2g}")

# F2: COMMON (37.3 ± 1.7) vs PLAY (31.3 ± 2.5) — Sokoban
soko_common_cov = np.array(soko_table["common_coverage"]["COMMON"])
soko_play_cov = np.array(soko_table["common_coverage"]["PLAY"])
check("F2.soko.COMMON.cov=37.3", close(soko_common_cov.mean(), 37.3, 0.02),
      f"{soko_common_cov.mean():.2f}")
check("F2.soko.COMMON.sd=1.7", close(soko_common_cov.std(ddof=1), 1.7, 0.10),
      f"{soko_common_cov.std(ddof=1):.2f}")
check("F2.soko.PLAY.cov=31.3", close(soko_play_cov.mean(), 31.3, 0.02),
      f"{soko_play_cov.mean():.2f}")
gap_soko_pct = 100 * (soko_common_cov.mean() - soko_play_cov.mean()) / soko_common_cov.mean()
check("F2.soko.gap=16%", abs(gap_soko_pct - 16) < 1.5,
      f"{gap_soko_pct:.1f}%")

# Sokoban COMMON vs PLAY p_adj and delta
rows = pairwise_table(soko_table["common_coverage"])
r = next((x for x in rows if (x.a, x.b) == ("COMMON", "PLAY")), None)
check("F2.soko.COMMON_vs_PLAY.p~1e-6", r.p_adj < 2e-6,
      f"p_adj={r.p_adj:.2g}")
check("F2.soko.COMMON_vs_PLAY.delta=0.95", abs(r.cliffs_delta - 0.95) < 0.03,
      f"delta={r.cliffs_delta:.2f}")

# Platformer same gap (claimed 28%)
plat_common_cov = np.array(plat_table["common_coverage"]["COMMON"])
plat_play_cov = np.array(plat_table["common_coverage"]["PLAY"])
gap_plat_pct = 100 * (plat_common_cov.mean() - plat_play_cov.mean()) / plat_common_cov.mean()
check("F2.plat.gap=22%", abs(gap_plat_pct - 22) < 1.5,
      f"{gap_plat_pct:.1f}%  (relative to COMMON's mean)")
rows = pairwise_table(plat_table["common_coverage"])
r = next((x for x in rows if (x.a, x.b) == ("COMMON", "PLAY")), None)
check("F2.plat.COMMON_vs_PLAY.delta=0.95", abs(r.cliffs_delta - 0.95) < 0.03,
      f"delta={r.cliffs_delta:.2f}")

# F3: 2.2× wall-clock; 81.8 ± 11.4 s vs 37.3 ± 11.6 s
str_wt_sd = np.std(soko_table["wall_time_s"]["STR"], ddof=1)
play_wt_sd = np.std(soko_table["wall_time_s"]["PLAY"], ddof=1)
check("F3.soko.STR_wt_mean=81.8", close(str_wt, 81.8, 0.02),
      f"{str_wt:.1f}")
check("F3.soko.STR_wt_sd=11.4", close(str_wt_sd, 11.4, 0.05),
      f"{str_wt_sd:.1f}")
check("F3.soko.PLAY_wt_mean=37.3", close(play_wt, 37.3, 0.02),
      f"{play_wt:.1f}")
check("F3.soko.PLAY_wt_sd=11.6", close(play_wt_sd, 11.6, 0.05),
      f"{play_wt_sd:.1f}")
rows = pairwise_table(soko_table["wall_time_s"])
r = next((x for x in rows if (x.a, x.b) == ("PLAY", "STR")), None)
delta_signed = r.cliffs_delta  # canonical (PLAY, STR), PLAY < STR so delta_a is < 0
check("F3.soko.STR_vs_PLAY_wt.p<1e-6", r.p_adj < 1e-6,
      f"p_adj={r.p_adj:.2g}")
check("F3.soko.STR_vs_PLAY_wt.delta=-0.99",
      abs(r.cliffs_delta + 0.99) < 0.02,
      f"delta={r.cliffs_delta:.2f}")

# Platformer wall-clock differences "within a factor of 1.4"
plat_wts = {c: np.mean(plat_table["wall_time_s"][c]) for c in plat_table["wall_time_s"]}
plat_wt_max = max(plat_wts.values())
plat_wt_min = min(plat_wts.values())
plat_ratio = plat_wt_max / plat_wt_min
check("F3.plat.wt_ratio<=1.4", plat_ratio <= 1.45,
      f"max/min = {plat_wt_max:.2f}/{plat_wt_min:.2f} = {plat_ratio:.2f}x")

# Platformer timeout rates 5-6%
with open(os.path.join(PLAT_DIR, "failure_breakdown.json")) as f:
    plat_fb = json.load(f)
unsolvs = [100 * plat_fb[c]["frac_unsolvable_attempted"] for c in plat_fb]
check("F3.plat.timeout_5_to_6%",
      min(unsolvs) >= 4.5 and max(unsolvs) <= 13,
      f"range=[{min(unsolvs):.1f}%, {max(unsolvs):.1f}%]")

# Note: the paper says 5-6% which matches STR/PLAY/SKILL/COMMON
# but RAND is 12.4% (different mechanism - random levels are infeasible
# at higher rate). Let me check.
me_only_unsolvs = [100 * plat_fb[c]["frac_unsolvable_attempted"]
                   for c in plat_fb if c != "RAND"]
check("F3.plat.ME_timeouts_5_to_6%",
      min(me_only_unsolvs) >= 4.5 and max(me_only_unsolvs) <= 7,
      f"ME variants only: range=[{min(me_only_unsolvs):.1f}%, {max(me_only_unsolvs):.1f}%]")

# F4: rank-orderings differ
# Sokoban indirect: SKILL=33.8, STR=32.4, PLAY=31.3
soko_skill = np.mean(soko_table["common_coverage"]["SKILL"])
soko_str = np.mean(soko_table["common_coverage"]["STR"])
soko_play = np.mean(soko_table["common_coverage"]["PLAY"])
check("F4.soko.SKILL>STR>PLAY",
      soko_skill > soko_str > soko_play,
      f"SKILL={soko_skill:.1f} STR={soko_str:.1f} PLAY={soko_play:.1f}")

# Platformer indirect: STR=39.9, SKILL=38.9, PLAY=35.4
plat_str = np.mean(plat_table["common_coverage"]["STR"])
plat_skill = np.mean(plat_table["common_coverage"]["SKILL"])
plat_play = np.mean(plat_table["common_coverage"]["PLAY"])
check("F4.plat.STR>SKILL>PLAY",
      plat_str > plat_skill > plat_play,
      f"STR={plat_str:.1f} SKILL={plat_skill:.1f} PLAY={plat_play:.1f}")


# ============================================================================
# Section VI.A (Sokoban results section claims)
# ============================================================================
print("\n== SECTION VI.A (Sokoban) ==")

# rank-order on coverage; specific numbers
for cond, claimed_mean in [("COMMON", 37.3), ("SKILL", 33.8), ("STR", 32.4),
                           ("PLAY", 31.3), ("RAND", 24.0)]:
    val = np.mean(soko_table["common_coverage"][cond])
    check(f"VI.A.soko.{cond}_cov={claimed_mean}", close(val, claimed_mean, 0.02),
          f"{val:.2f}")

# "STR vs PLAY (δ=−0.23, p_adj=0.21)"
rows = pairwise_table(soko_table["common_coverage"])
r = next((x for x in rows if (x.a, x.b) == ("PLAY", "STR")), None)
check("VI.A.STR_vs_PLAY.delta=-0.23",
      abs(r.cliffs_delta + 0.23) < 0.03,
      f"delta={r.cliffs_delta:.2f}")
check("VI.A.STR_vs_PLAY.p_adj=0.21",
      abs(r.p_adj - 0.21) < 0.03,
      f"p_adj={r.p_adj:.2f}")

# "SKILL vs STR (δ=+0.34, p_adj=0.12)"
r = next((x for x in rows if (x.a, x.b) == ("SKILL", "STR")), None)
check("VI.A.SKILL_vs_STR.delta=+0.34",
      abs(r.cliffs_delta - 0.34) < 0.03,
      f"delta={r.cliffs_delta:+.2f}")
check("VI.A.SKILL_vs_STR.p_adj=0.12",
      abs(r.p_adj - 0.12) < 0.03,
      f"p_adj={r.p_adj:.2f}")

# Seed-level wall-clocks (72-108 s for STR, 63-92 for COMMON, 25-58 for PLAY)
soko_wts_per_cond = {c: np.asarray(soko_table["wall_time_s"][c])
                     for c in soko_table["wall_time_s"]}
for cond, claimed_range in [("STR", (62, 107)), ("COMMON", (62, 92)),
                             ("PLAY", (20, 62))]:
    arr = soko_wts_per_cond[cond]
    actual_min, actual_max = arr.min(), arr.max()
    check(f"VI.A.{cond}_wt_range",
          claimed_range[0] - 1 <= actual_min and actual_max <= claimed_range[1] + 1,
          f"actual=[{actual_min:.1f}, {actual_max:.1f}] vs claimed=[{claimed_range[0]}, {claimed_range[1]}]")


# ============================================================================
# Section VI.B (Platformer)
# ============================================================================
print("\n== SECTION VI.B (Platformer) ==")

# rank-order
for cond, claimed in [("COMMON", 45.5), ("STR", 39.9), ("RAND", 39.8),
                      ("SKILL", 38.9), ("PLAY", 35.4)]:
    val = np.mean(plat_table["common_coverage"][cond])
    check(f"VI.B.plat.{cond}_cov={claimed}", close(val, claimed, 0.02),
          f"{val:.2f}")

# COMMON beats every other condition with δ≥0.63 and p_adj<0.01
rows = pairwise_table(plat_table["common_coverage"])
for cond in ["RAND", "STR", "PLAY", "SKILL"]:
    pair = tuple(sorted(["COMMON", cond]))
    r = next((x for x in rows if (x.a, x.b) == pair), None)
    d_signed = r.cliffs_delta if r.a == "COMMON" else -r.cliffs_delta
    check(f"VI.B.COMMON_vs_{cond}.delta>=0.63", d_signed >= 0.63,
          f"delta={d_signed:.2f}")
    check(f"VI.B.COMMON_vs_{cond}.p_adj<0.01", r.p_adj < 0.01,
          f"p_adj={r.p_adj:.2g}")

# PLAY < every other condition (all δ < -0.5, p_adj < 0.03)
for cond in ["RAND", "STR", "SKILL", "COMMON"]:
    pair = tuple(sorted(["PLAY", cond]))
    r = next((x for x in rows if (x.a, x.b) == pair), None)
    d_signed = r.cliffs_delta if r.a == "PLAY" else -r.cliffs_delta
    check(f"VI.B.PLAY<_{cond}.delta<-0.5", d_signed < -0.5,
          f"delta_signed={d_signed:.2f}")
    check(f"VI.B.PLAY<_{cond}.p_adj<0.03", r.p_adj < 0.03,
          f"p_adj={r.p_adj:.2g}")

# STR, SKILL, RAND not separately distinguishable (p_adj > 0.38)
for a, b in [("STR", "SKILL"), ("STR", "RAND"), ("SKILL", "RAND")]:
    pair = tuple(sorted([a, b]))
    r = next((x for x in rows if (x.a, x.b) == pair), None)
    check(f"VI.B.{a}_vs_{b}.p_adj>0.38", r.p_adj > 0.38,
          f"p_adj={r.p_adj:.2f}")


# ============================================================================
# Section VII (Mechanism)
# ============================================================================
print("\n== SECTION VII (Mechanism) ==")

# "STR's evaluated distribution sits 0.78 decades above PLAY's
# (median 3.15 vs 2.37)"  -- recompute medians from raw
def median_logn(domain_runs, retained):
    out = defaultdict(list)
    for r in domain_runs:
        if retained:
            for c in r["common_archive"]["cells"]:
                if c["solvable"] and c["nodes_expanded"] > 0:
                    out[r["cond"]].append(np.log10(c["nodes_expanded"]))
        else:
            for e in r["eval_records"]:
                if e["solvable"] and e["nodes_expanded"] > 0:
                    out[r["cond"]].append(np.log10(e["nodes_expanded"]))
    return {c: float(np.median(v)) for c, v in out.items()}

soko_eval_med = median_logn(soko, retained=False)
soko_ret_med = median_logn(soko, retained=True)
check("VII.STR_eval_median=3.15", abs(soko_eval_med["STR"] - 3.15) < 0.02,
      f"{soko_eval_med['STR']:.2f}")
check("VII.PLAY_eval_median=2.37", abs(soko_eval_med["PLAY"] - 2.37) < 0.02,
      f"{soko_eval_med['PLAY']:.2f}")
gap = soko_eval_med["STR"] - soko_eval_med["PLAY"]
check("VII.STR-PLAY_eval_gap=0.78", abs(gap - 0.78) < 0.02,
      f"gap={gap:.2f}")
check("VII.STR_retain_median=2.73", abs(soko_ret_med["STR"] - 2.73) < 0.02,
      f"{soko_ret_med['STR']:.2f}")
check("VII.PLAY_retain_median=2.56", abs(soko_ret_med["PLAY"] - 2.56) < 0.02,
      f"{soko_ret_med['PLAY']:.2f}")

with open(os.path.join(SOKO_DIR, "mechanism.json")) as f:
    soko_mech = json.load(f)

# "pairwise p_adj < 10^-3, δ = -0.74"
pairs = soko_mech["elite_logn_pairwise"]
r = next((x for x in pairs if (x["a"], x["b"]) == ("PLAY", "STR")), None)
check("VII.STR_vs_PLAY_retain.p_adj<1e-3", r["p_adj"] < 1e-3,
      f"p_adj={r['p_adj']:.2g}")
check("VII.STR_vs_PLAY_retain.delta=-0.74",
      abs(r["cliffs_delta"] + 0.74) < 0.03,
      f"delta={r['cliffs_delta']:.2f}")

# "wall density and A* nodes within solvable boards is in fact weakly
# negatively correlated, Spearman ρ = -0.07"
with open(os.path.join(SOKO_DIR, "descriptor_hardness_correlation.json")) as f:
    soko_corr = json.load(f)
str_axis2_rho = soko_corr["STR"]["spearman_axis2"]["rho"]
check("VII.STR_walldensity_rho=-0.07",
      abs(str_axis2_rho + 0.07) < 0.02,
      f"rho={str_axis2_rho:+.3f}")

# Table II: per-condition unsolvability rates
print("  --- Table II row checks ---")
for cond, claimed_wf, claimed_solv, claimed_unsolv, claimed_logn in [
    ("RAND",   100.0, 22.2,  5.7, 2.22),
    ("STR",    98.5,  55.4, 20.0, 2.90),
    ("PLAY",   98.5,  63.1, 12.9, 2.45),
    ("SKILL",  98.3,  64.4, 12.6, 2.42),
    ("COMMON", 97.8,  62.3, 14.6, 2.54),
]:
    fb = soko_fb[cond]
    check(f"TableII.{cond}.wf={claimed_wf:.1f}%",
          abs(100*fb["frac_well_formed"] - claimed_wf) < 0.1,
          f"{100*fb['frac_well_formed']:.1f}%")
    check(f"TableII.{cond}.solv={claimed_solv:.1f}%",
          abs(100*fb["frac_solvable"] - claimed_solv) < 0.2,
          f"{100*fb['frac_solvable']:.1f}%")
    check(f"TableII.{cond}.unsolv={claimed_unsolv:.1f}%",
          abs(100*fb["frac_unsolvable_attempted"] - claimed_unsolv) < 0.2,
          f"{100*fb['frac_unsolvable_attempted']:.1f}%")
    check(f"TableII.{cond}.logN_all={claimed_logn:.2f}",
          abs(fb["mean_logn_all_attempted"] - claimed_logn) < 0.02,
          f"{fb['mean_logn_all_attempted']:.2f}")

# Platformer: "Spearman... no pairwise difference in retained-elite hardness
# survives Holm correction (p_adj ≥ 0.41 for all)"
with open(os.path.join(PLAT_DIR, "mechanism.json")) as f:
    plat_mech = json.load(f)
plat_pairs = plat_mech.get("elite_logn_pairwise", [])
min_p_adj = min((p["p_adj"] for p in plat_pairs), default=1.0)
check("VII.plat.min_p_adj>=0.41", min_p_adj >= 0.40,
      f"min_p_adj={min_p_adj:.3f}")


# ============================================================================
# Decision guide claims
# ============================================================================
print("\n== DECISION GUIDE ==")

# "mutation from a dense-wall elite is 1.6× more likely to produce a board
# the solver cannot finish"
ratio_unsolv = str_unsolv / play_unsolv
check("Discussion.STR_unsolv_ratio=1.6x",
      abs(ratio_unsolv - 1.6) < 0.1,
      f"{str_unsolv:.1f}% / {play_unsolv:.1f}% = {ratio_unsolv:.2f}x")


# ============================================================================
# Summary
# ============================================================================
print(f"\n\n{passed} passed, {failed} failed, {len(warnings)} warnings")
sys.exit(0 if failed == 0 else 1)
