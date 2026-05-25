"""Statistical utilities: Mann-Whitney U, Cliff's delta, Holm-Bonferroni,
bootstrap CIs. No fancy dependencies — numpy + scipy only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass
class PairwiseStat:
    a: str
    b: str
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    p_value: float
    p_adj: float
    cliffs_delta: float
    interpretation: str   # 'negligible' | 'small' | 'medium' | 'large'
    direction: str        # '>' or '<' or '=' for mean_a vs mean_b


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), with ties counted as zero.

    Bounded in [-1, 1]. Returns 0.0 when a == b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    # Vectorised pairwise comparisons.
    diff = a[:, None] - b[None, :]
    gt = (diff > 0).sum()
    lt = (diff < 0).sum()
    return float((gt - lt) / (n_a * n_b))


def cliffs_interpretation(d: float) -> str:
    """Romano et al. (2006) thresholds."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Step-down Holm correction. Returns adjusted p-values in original order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, i in enumerate(order):
        v = (n - rank) * p[i]
        if v > 1.0:
            v = 1.0
        running_max = max(running_max, v)
        adj[i] = running_max
    return adj.tolist()


def pairwise_table(groups: dict, metric_label: str = "metric") -> List[PairwiseStat]:
    """Compute all pairwise Mann-Whitney + Cliff's delta tests between
    every (unordered) pair of groups. Apply Holm-Bonferroni across the
    whole family. Returns a list of PairwiseStat in canonical order
    (alphabetical by (a, b) with a < b).
    """
    names = sorted(groups.keys())
    raw_p = []
    rows: List[PairwiseStat] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa, xb = np.asarray(groups[a]), np.asarray(groups[b])
            # two-sided
            stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
            d = cliffs_delta(xa, xb)
            direction = ">" if np.mean(xa) > np.mean(xb) else \
                        "<" if np.mean(xa) < np.mean(xb) else "="
            row = PairwiseStat(
                a=a, b=b,
                mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
                median_a=float(np.median(xa)), median_b=float(np.median(xb)),
                p_value=float(p), p_adj=float(p),  # adjusted below
                cliffs_delta=d,
                interpretation=cliffs_interpretation(d),
                direction=direction,
            )
            rows.append(row)
            raw_p.append(p)
    adj = holm_bonferroni(raw_p)
    for row, p_adj in zip(rows, adj):
        row.p_adj = float(p_adj)
    return rows


def bootstrap_ci(x: Sequence[float], n_boot: int = 5000,
                 alpha: float = 0.05, statistic=np.mean,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic. Returns (point, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[k] = statistic(x[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(statistic(x)), lo, hi


def stouffer_combine(pvalues: Sequence[float],
                     weights: Sequence[float] = None) -> float:
    """Stouffer's Z-method for combining p-values (one-sided)."""
    from scipy.stats import norm
    p = np.asarray(pvalues, dtype=float)
    if weights is None:
        weights = np.ones_like(p)
    weights = np.asarray(weights, dtype=float)
    # Avoid p exactly 0 or 1.
    p = np.clip(p, 1e-15, 1 - 1e-15)
    z = norm.isf(p)
    z_combined = (weights * z).sum() / np.sqrt((weights ** 2).sum())
    return float(norm.sf(z_combined))


def spearman_with_ci(x: Sequence[float], y: Sequence[float],
                     n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float, float]:
    """Spearman rho with bootstrap 95% CI and p-value."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boots[k] = r if not np.isnan(r) else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi), float(p)
