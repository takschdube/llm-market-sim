# src/analysis/statistics.py
"""
Statistical Analysis for Market Experiments
============================================
Functions for rigorous statistical comparison of experimental conditions.

Designed for MABS-style multi-agent simulation papers with:
- Non-parametric tests (robust to non-normality)
- Effect size calculations
- Bootstrap confidence intervals
- Multiple comparison corrections
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two conditions."""
    condition_a: str
    condition_b: str
    metric: str

    # Descriptive statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int

    # Hypothesis test
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At alpha=0.05

    # Effect size
    effect_size: float
    effect_size_name: str
    effect_interpretation: str  # "small", "medium", "large"

    # Confidence interval for difference
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"{self.condition_a} vs {self.condition_b} ({self.metric}):\n"
            f"  {self.condition_a}: {self.mean_a:.3f} ± {self.std_a:.3f} (n={self.n_a})\n"
            f"  {self.condition_b}: {self.mean_b:.3f} ± {self.std_b:.3f} (n={self.n_b})\n"
            f"  {self.test_name}: statistic={self.statistic:.3f}, p={self.p_value:.4f}{sig}\n"
            f"  {self.effect_size_name}={self.effect_size:.3f} ({self.effect_interpretation})\n"
            f"  95% CI for difference: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
        )


def check_normality(data: List[float], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Test for normality using Shapiro-Wilk test.

    Returns:
        (is_normal, p_value)
    """
    if len(data) < 3:
        return False, 0.0

    stat, p_value = stats.shapiro(data)
    return p_value > alpha, p_value


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cliff's delta (non-parametric effect size).

    Interpretation:
        |d| < 0.147: negligible
        0.147 <= |d| < 0.33: small
        0.33 <= |d| < 0.474: medium
        |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)

    # Count dominance
    more = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)

    return (more - less) / (n1 * n2)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    delta = abs(delta)
    if delta < 0.147:
        return "negligible"
    elif delta < 0.33:
        return "small"
    elif delta < 0.474:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    data: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Sample data
        statistic: "mean" or "median"
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        seed: Random seed for reproducibility

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        else:
            bootstrap_stats.append(np.median(sample))

    # Calculate percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def bootstrap_difference_ci(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap CI for difference in means.
    """
    rng = np.random.RandomState(seed)
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)

    differences = []
    for _ in range(n_bootstrap):
        s1 = rng.choice(g1, size=n1, replace=True)
        s2 = rng.choice(g2, size=n2, replace=True)
        differences.append(np.mean(s1) - np.mean(s2))

    alpha = 1 - confidence
    lower = np.percentile(differences, 100 * alpha / 2)
    upper = np.percentile(differences, 100 * (1 - alpha / 2))

    return lower, upper


def compare_two_conditions(
    data_a: List[float],
    data_b: List[float],
    name_a: str,
    name_b: str,
    metric_name: str,
    use_parametric: Optional[bool] = None,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Compare two experimental conditions with appropriate statistical tests.

    Automatically selects parametric (t-test) or non-parametric (Mann-Whitney U)
    based on normality of data, unless specified.

    Args:
        data_a: Observations from condition A
        data_b: Observations from condition B
        name_a: Name of condition A
        name_b: Name of condition B
        metric_name: Name of the metric being compared
        use_parametric: Force parametric (True) or non-parametric (False) test
        alpha: Significance level

    Returns:
        ComparisonResult with full statistical analysis
    """
    # Descriptive stats
    mean_a, mean_b = np.mean(data_a), np.mean(data_b)
    std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
    n_a, n_b = len(data_a), len(data_b)

    # Check normality if not specified
    if use_parametric is None:
        normal_a, _ = check_normality(data_a)
        normal_b, _ = check_normality(data_b)
        use_parametric = normal_a and normal_b

    # Hypothesis test
    if use_parametric:
        statistic, p_value = stats.ttest_ind(data_a, data_b)
        test_name = "Independent t-test"
        effect = cohens_d(data_a, data_b)
        effect_name = "Cohen's d"
        effect_interp = interpret_cohens_d(effect)
    else:
        statistic, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        test_name = "Mann-Whitney U"
        effect = cliffs_delta(data_a, data_b)
        effect_name = "Cliff's delta"
        effect_interp = interpret_cliffs_delta(effect)

    # Bootstrap CI for difference
    ci_lower, ci_upper = bootstrap_difference_ci(data_a, data_b)

    return ComparisonResult(
        condition_a=name_a,
        condition_b=name_b,
        metric=metric_name,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        n_a=n_a,
        n_b=n_b,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=effect,
        effect_size_name=effect_name,
        effect_interpretation=effect_interp,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compare_multiple_conditions(
    data: Dict[str, List[float]],
    metric_name: str,
    alpha: float = 0.05
) -> Dict:
    """
    Compare 3+ conditions using Kruskal-Wallis with post-hoc Dunn's test.

    Args:
        data: Dict mapping condition name to list of observations
        metric_name: Name of the metric
        alpha: Significance level

    Returns:
        Dict with omnibus test and pairwise comparisons
    """
    conditions = list(data.keys())
    groups = [data[c] for c in conditions]

    # Omnibus test
    h_stat, p_omnibus = stats.kruskal(*groups)

    result = {
        "metric": metric_name,
        "omnibus_test": "Kruskal-Wallis H",
        "h_statistic": h_stat,
        "p_omnibus": p_omnibus,
        "omnibus_significant": p_omnibus < alpha,
        "conditions": conditions,
        "descriptive": {
            c: {"mean": np.mean(data[c]), "std": np.std(data[c], ddof=1), "n": len(data[c])}
            for c in conditions
        },
        "pairwise": [],
    }

    # Post-hoc pairwise comparisons (if omnibus significant)
    if p_omnibus < alpha:
        # Bonferroni correction
        n_comparisons = len(conditions) * (len(conditions) - 1) // 2
        corrected_alpha = alpha / n_comparisons

        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1:]:
                comparison = compare_two_conditions(
                    data[c1], data[c2], c1, c2, metric_name,
                    use_parametric=False, alpha=corrected_alpha
                )
                result["pairwise"].append({
                    "comparison": f"{c1} vs {c2}",
                    "p_value": comparison.p_value,
                    "significant_corrected": comparison.p_value < corrected_alpha,
                    "effect_size": comparison.effect_size,
                    "effect_interpretation": comparison.effect_interpretation,
                })

    return result


def summarize_experiment(
    results_by_condition: Dict[str, List[Dict]],
    metrics: List[str] = None
) -> Dict:
    """
    Summarize experimental results across conditions.

    Args:
        results_by_condition: Dict mapping condition name to list of run results
        metrics: List of metric names to summarize (default: convergence, efficiency)

    Returns:
        Summary statistics for each condition and metric
    """
    if metrics is None:
        metrics = ["convergence_mad", "avg_price", "total_trades"]

    summary = {}

    for condition, runs in results_by_condition.items():
        summary[condition] = {
            "n_runs": len(runs),
            "metrics": {}
        }

        for metric in metrics:
            values = [r.get(metric) for r in runs if r.get(metric) is not None]
            if values:
                ci_lower, ci_upper = bootstrap_ci(values)
                summary[condition]["metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),
                    "median": np.median(values),
                    "min": min(values),
                    "max": max(values),
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                }

    return summary


def power_analysis_two_sample(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0
) -> int:
    """
    Calculate required sample size per group for two-sample t-test.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired statistical power
        ratio: Ratio of n2/n1 (default 1.0 for equal groups)

    Returns:
        Required sample size per group
    """
    from scipy.stats import norm

    # Z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    # Sample size formula for two-sample t-test
    n = ((z_alpha + z_power) ** 2 * (1 + 1 / ratio) * 2) / (effect_size ** 2)

    return int(np.ceil(n))
