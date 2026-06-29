# src/analysis/icc.py
"""Intraclass correlation for cognitive monoculture analysis.

This module provides the canonical estimator for the architectural correlation
parameter rho_A defined in Theorem 1 of Dube (2026). The estimator computes
the intraclass correlation of same-role price deviations within a single
LLM provider, averaged over rounds and replications, and returns a point
estimate, a finite-sample standard error, and a normal-approximation 95%
confidence interval.

The legacy spread-based proxy in metrics.compute_cross_sectional_correlation
remains available but is superseded by the estimator here for any analysis
that needs to ground rho_A and the critical threshold c* = 1/sqrt(rho_A * n).

References
----------
Dube, T. (2026). Cognitive Monoculture in AI-Populated Markets. Theorem 1.
Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological Bulletin, 86(2), 420-428.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ICCResult:
    """Output of an intraclass-correlation estimator.

    Attributes
    ----------
    rho_hat : float
        Point estimate of the intraclass correlation in [-1, 1].
    se : float
        Finite-sample standard error of the point estimate.
    ci95 : tuple[float, float]
        Lower and upper 95 percent confidence bounds (normal approximation).
    n_pairs : int
        Number of unique same-role pairs that entered the estimate.
    n_rounds : int
        Number of rounds in which a same-role pair contributed.
    """

    rho_hat: float
    se: float
    ci95: Tuple[float, float]
    n_pairs: int
    n_rounds: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "rho_hat": self.rho_hat,
            "se": self.se,
            "ci_lo": self.ci95[0],
            "ci_hi": self.ci95[1],
            "n_pairs": self.n_pairs,
            "n_rounds": self.n_rounds,
        }


def _deviations_by_agent(
    round_data: Sequence[Dict],
    agent_roles: Dict[str, str],
    equilibrium_price: float,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Build a same-shape deviation matrix per agent across rounds.

    Returns a dict mapping agent_id to a 1D ndarray of length n_rounds, with
    NaN for rounds in which the agent did not submit a price. The list of
    rounds is returned in submission order.
    """
    rounds = [r.get("round", idx) for idx, r in enumerate(round_data)]
    agent_ids = sorted(agent_roles.keys())
    dev = {aid: np.full(len(round_data), np.nan, dtype=float) for aid in agent_ids}

    for t, round_info in enumerate(round_data):
        decisions = round_info.get("decisions", {})
        for aid, d in decisions.items():
            price = d.get("price") if isinstance(d, dict) else None
            if price is None and isinstance(d, dict):
                price = d.get("decision", {}).get("price")
            if price is None:
                continue
            if aid in dev:
                dev[aid][t] = float(price) - equilibrium_price

    return dev, [str(r) for r in rounds]


def _pairwise_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson correlation over rounds where both agents acted."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return None
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def estimate_within_model_icc(
    round_data: Sequence[Dict],
    agent_models: Dict[str, str],
    agent_roles: Dict[str, str],
    equilibrium_price: float,
    model: Optional[str] = None,
) -> ICCResult:
    """Estimate rho_within for one architecture across same-role agents.

    Implementation. The estimator is the unweighted mean of pairwise Pearson
    correlations of round-level price deviations, taken over all distinct
    same-architecture same-role agent pairs that share at least three rounds
    of joint observations. The standard error is the empirical standard error
    of the pairwise correlations across pairs, used as a finite-sample proxy
    for the standard error of the mean ICC estimator. The 95 percent
    confidence interval is the normal-approximation interval
    rho_hat +/- 1.96 * se.

    Parameters
    ----------
    round_data : sequence of round dicts
        Each round dict carries a 'decisions' dict mapping agent_id to a
        decision dict with a numeric 'price' field, either at top level or
        nested under 'decision'.
    agent_models : dict[str, str]
        Maps agent_id to architecture or provider label.
    agent_roles : dict[str, str]
        Maps agent_id to 'buyer' or 'seller'.
    equilibrium_price : float
        Walrasian price p* used to compute deviations.
    model : str, optional
        If given, restricts the estimate to agents whose label equals model.
        If omitted, the estimator pools across all architectures present.

    Returns
    -------
    ICCResult
    """
    targets = {
        aid: agent_models[aid]
        for aid in agent_models
        if model is None or agent_models[aid] == model
    }
    sub_roles = {aid: agent_roles.get(aid, "unknown") for aid in targets}

    dev_by_agent, _ = _deviations_by_agent(round_data, sub_roles, equilibrium_price)

    correlations: List[float] = []
    contributing_rounds = 0
    pairs = list(combinations(sorted(targets.keys()), 2))
    for a, b in pairs:
        if agent_models[a] != agent_models[b]:
            continue
        if sub_roles[a] != sub_roles[b]:
            continue
        rho = _pairwise_pearson(dev_by_agent[a], dev_by_agent[b])
        if rho is None:
            continue
        correlations.append(rho)
        contributing_rounds = max(
            contributing_rounds,
            int((~np.isnan(dev_by_agent[a]) & ~np.isnan(dev_by_agent[b])).sum()),
        )

    if not correlations:
        return ICCResult(
            rho_hat=float("nan"),
            se=float("nan"),
            ci95=(float("nan"), float("nan")),
            n_pairs=0,
            n_rounds=0,
        )

    rho_hat = float(np.mean(correlations))
    if len(correlations) > 1:
        se = float(np.std(correlations, ddof=1) / sqrt(len(correlations)))
    else:
        se = float("nan")
    ci_lo = rho_hat - 1.96 * se if se == se else rho_hat
    ci_hi = rho_hat + 1.96 * se if se == se else rho_hat
    return ICCResult(
        rho_hat=rho_hat,
        se=se,
        ci95=(ci_lo, ci_hi),
        n_pairs=len(correlations),
        n_rounds=contributing_rounds,
    )


def estimate_between_model_icc(
    round_data: Sequence[Dict],
    agent_models: Dict[str, str],
    agent_roles: Dict[str, str],
    equilibrium_price: float,
) -> ICCResult:
    """Estimate rho_between across pairs of different architectures.

    Same statistic as estimate_within_model_icc but restricted to pairs whose
    architecture labels differ and whose roles match.
    """
    dev_by_agent, _ = _deviations_by_agent(round_data, agent_roles, equilibrium_price)

    correlations: List[float] = []
    contributing_rounds = 0
    pairs = list(combinations(sorted(agent_models.keys()), 2))
    for a, b in pairs:
        if agent_models[a] == agent_models[b]:
            continue
        if agent_roles.get(a) != agent_roles.get(b):
            continue
        rho = _pairwise_pearson(dev_by_agent[a], dev_by_agent[b])
        if rho is None:
            continue
        correlations.append(rho)
        contributing_rounds = max(
            contributing_rounds,
            int((~np.isnan(dev_by_agent[a]) & ~np.isnan(dev_by_agent[b])).sum()),
        )

    if not correlations:
        return ICCResult(
            rho_hat=float("nan"),
            se=float("nan"),
            ci95=(float("nan"), float("nan")),
            n_pairs=0,
            n_rounds=0,
        )

    rho_hat = float(np.mean(correlations))
    se = (
        float(np.std(correlations, ddof=1) / sqrt(len(correlations)))
        if len(correlations) > 1
        else float("nan")
    )
    ci_lo = rho_hat - 1.96 * se if se == se else rho_hat
    ci_hi = rho_hat + 1.96 * se if se == se else rho_hat
    return ICCResult(
        rho_hat=rho_hat,
        se=se,
        ci95=(ci_lo, ci_hi),
        n_pairs=len(correlations),
        n_rounds=contributing_rounds,
    )


def critical_concentration(rho_a: float, n: int) -> float:
    """Compute c* = 1 / sqrt(rho_A * n), the critical concentration threshold.

    For rho_a in (0, 1] and n >= 1, returns the threshold at which LLM
    correlation dominates aggregate variance under Theorem 2 of Dube (2026).
    Returns float('inf') for rho_a <= 0 (no monoculture, no finite threshold).
    """
    if rho_a is None or rho_a <= 0 or not np.isfinite(rho_a):
        return float("inf")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return float(1.0 / sqrt(rho_a * n))


def critical_concentration_band(icc: ICCResult, n: int) -> Tuple[float, float, float]:
    """Propagate an ICC interval to a critical-threshold interval.

    Returns (c_hat, c_lower, c_upper) where c_hat = c*(rho_hat) and the
    bounds use the upper and lower confidence limits of rho_A. The bounds
    are clipped to [0, 1] since concentration is a fraction.
    """
    c_hat = critical_concentration(icc.rho_hat, n)
    rho_lo = max(icc.ci95[0], 1e-9)
    rho_hi = min(icc.ci95[1], 1.0)
    c_upper = critical_concentration(rho_lo, n)  # smaller rho -> larger c*
    c_lower = critical_concentration(rho_hi, n)
    return (
        min(max(c_hat, 0.0), 1.0) if np.isfinite(c_hat) else c_hat,
        min(max(c_lower, 0.0), 1.0) if np.isfinite(c_lower) else c_lower,
        min(max(c_upper, 0.0), 1.0) if np.isfinite(c_upper) else c_upper,
    )


def monoculture_effect(within: ICCResult, between: ICCResult) -> Dict[str, float]:
    """Compute Delta rho = rho_within - rho_between with a pooled SE.

    Returns a dict with the point estimate, a pooled standard error
    (sqrt(se_w^2 + se_b^2)), and a normal-approximation p-value for the
    null Delta rho = 0.
    """
    if not (np.isfinite(within.rho_hat) and np.isfinite(between.rho_hat)):
        return {
            "delta_rho": float("nan"),
            "se": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
        }
    delta = within.rho_hat - between.rho_hat
    se = sqrt(
        (within.se if np.isfinite(within.se) else 0.0) ** 2
        + (between.se if np.isfinite(between.se) else 0.0) ** 2
    )
    if se < 1e-12:
        return {"delta_rho": delta, "se": se, "z": float("inf"), "p_value": 0.0}
    z = delta / se
    from math import erf

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return {"delta_rho": delta, "se": se, "z": z, "p_value": p}
