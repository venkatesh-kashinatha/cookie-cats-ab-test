"""Statistics for a two-group A/B test on binary and skewed numeric metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> dict:
    """Sample ratio mismatch: chi-square goodness-of-fit test of the group sizes."""
    total = n_a + n_b
    expected = np.array([total * expected_share_a, total * (1 - expected_share_a)])
    chi2, p = stats.chisquare([n_a, n_b], expected)
    return {"n_a": n_a, "n_b": n_b, "share_a": n_a / total, "chi2": float(chi2), "p_value": float(p)}


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int) -> dict:
    """Two-sided z-test for p_b - p_a (pooled SE) with a 95% Wald CI (unpooled SE)."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    z = diff / se0
    p_value = 2 * stats.norm.sf(abs(z))
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return {"p_a": p_a, "p_b": p_b, "diff": diff, "relative_lift": diff / p_a, "z": z,
            "p_value": float(p_value), "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se}


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42,
                   stat=np.mean) -> dict:
    """Bootstrap the difference stat(b) - stat(a) by resampling players with replacement.

    For 0/1 arrays the resample mean is drawn from a binomial, which is exactly equivalent
    to resampling rows and much faster.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    binary = stat is np.mean and set(np.unique(np.concatenate([a, b]))) <= {0, 1}
    if binary:
        boot_a = rng.binomial(len(a), a.mean(), n_boot) / len(a)
        boot_b = rng.binomial(len(b), b.mean(), n_boot) / len(b)
    else:
        boot_a = np.array([stat(rng.choice(a, len(a))) for _ in range(n_boot)])
        boot_b = np.array([stat(rng.choice(b, len(b))) for _ in range(n_boot)])
    diffs = boot_b - boot_a
    return {
        "diff": float(stat(b) - stat(a)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_b_better": float((diffs > 0).mean()),
        "boot_a": boot_a,
        "boot_b": boot_b,
    }


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"u": float(res.statistic), "p_value": float(res.pvalue)}


def minimum_detectable_effect(p: float, n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute difference in a proportion the test could detect with the given power."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
