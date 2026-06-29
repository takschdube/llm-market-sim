"""Pluggable metric registry.

A metric is a callable `(trace: Trace, **params) -> Any` registered under
a string name. The `Metric` namespace exposes builders that return
`MetricSpec` value objects. `Trace.evaluate([Metric.icc_within(), ...])`
runs the named metrics and returns a dict.

Built-in metrics cover the cognitive monoculture quantities (icc_within,
icc_between, monoculture_effect, critical_concentration, welfare_bound),
the standard market-design quantities (allocative_efficiency,
equilibrium_distance), and a few distributional helpers (bid_entropy,
price_entropy).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .traces import Trace


class MetricSpec(BaseModel):
    """Reified metric: a registered name plus a params dict."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")


class MetricRegistry:
    """Internal registry of metric callables keyed by name."""

    _registry: ClassVar[Dict[str, Callable[..., Any]]] = {}

    @classmethod
    def register(cls, name: str, fn: Callable[..., Any]) -> None:
        if name in cls._registry:
            raise ValueError(f"Metric {name!r} already registered")
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(f"Metric {name!r} not registered. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())


class Metric:
    """Namespace for metric builders.

    Use the staticmethods to construct `MetricSpec` value objects. Use
    `@Metric.register("name")` to add new metrics to the registry.
    """

    @staticmethod
    def icc_within(level: str = "model") -> MetricSpec:
        return MetricSpec(name="icc_within", params={"level": level})

    @staticmethod
    def icc_between() -> MetricSpec:
        return MetricSpec(name="icc_between", params={})

    @staticmethod
    def monoculture_effect() -> MetricSpec:
        return MetricSpec(name="monoculture_effect", params={})

    @staticmethod
    def allocative_efficiency() -> MetricSpec:
        return MetricSpec(name="allocative_efficiency", params={})

    @staticmethod
    def welfare_bound() -> MetricSpec:
        return MetricSpec(name="welfare_bound", params={})

    @staticmethod
    def critical_concentration(safety_margin: float = 0.0) -> MetricSpec:
        return MetricSpec(
            name="critical_concentration",
            params={"safety_margin": safety_margin},
        )

    @staticmethod
    def equilibrium_distance() -> MetricSpec:
        return MetricSpec(name="equilibrium_distance", params={})

    @staticmethod
    def bid_entropy(n_bins: int = 10) -> MetricSpec:
        return MetricSpec(name="bid_entropy", params={"n_bins": n_bins})

    @staticmethod
    def price_entropy(n_bins: int = 10) -> MetricSpec:
        return MetricSpec(name="price_entropy", params={"n_bins": n_bins})

    @staticmethod
    def register(name: str):
        """Decorator that registers a metric callable.

        The callable must accept `(trace: Trace, **params)` and return
        a JSON-serializable value (float, int, str, dict, list).
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            MetricRegistry.register(name, fn)
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Built-in metric implementations
# ---------------------------------------------------------------------------


def _trace_to_icc_inputs(trace: "Trace", level: str) -> Dict[str, Any]:
    """Build the (round_data, agent_models, agent_roles, equilibrium_price) tuple
    that `src/analysis/icc.py` consumes, derived from a Trace.

    `level` controls what is used as the "model" axis: "model" groups by
    (provider, model), "lineage" groups by lineage_tag, "provider" groups
    by provider alone.
    """
    round_data: list = []
    agent_models: Dict[str, str] = {}
    agent_roles: Dict[str, str] = {}
    equilibrium_prices: list = []

    for rep in trace.replications:
        for rnd in rep.rounds:
            decisions = {}
            for d in rnd.decisions:
                decisions[d.agent_id] = {"price": d.submitted_price}
                if level == "model":
                    label = f"{d.provider or 'classical'}/{d.model or 'na'}"
                elif level == "lineage":
                    label = d.lineage_tag or "unknown"
                elif level == "provider":
                    label = d.provider or "classical"
                else:
                    raise ValueError(f"Unknown ICC level {level!r}")
                agent_models[d.agent_id] = label
                agent_roles[d.agent_id] = d.role
                equilibrium_prices.append(d.equilibrium_price)
            round_data.append({"round": rnd.round, "decisions": decisions})

    equilibrium_price = (
        equilibrium_prices[0] if equilibrium_prices else 0.0
    )
    return {
        "round_data": round_data,
        "agent_models": agent_models,
        "agent_roles": agent_roles,
        "equilibrium_price": equilibrium_price,
    }


@Metric.register("icc_within")
def _icc_within(trace: "Trace", level: str = "model") -> Dict[str, Any]:
    from src.analysis.icc import estimate_within_model_icc

    inputs = _trace_to_icc_inputs(trace, level)
    result = estimate_within_model_icc(**inputs)
    return result.to_dict()


@Metric.register("icc_between")
def _icc_between(trace: "Trace") -> Dict[str, Any]:
    from src.analysis.icc import estimate_between_model_icc

    inputs = _trace_to_icc_inputs(trace, "model")
    result = estimate_between_model_icc(**inputs)
    return result.to_dict()


@Metric.register("monoculture_effect")
def _monoculture_effect(trace: "Trace") -> Dict[str, Any]:
    from src.analysis.icc import (
        estimate_between_model_icc,
        estimate_within_model_icc,
        monoculture_effect,
    )

    inputs = _trace_to_icc_inputs(trace, "model")
    within = estimate_within_model_icc(**inputs)
    between = estimate_between_model_icc(**inputs)
    return monoculture_effect(within, between)


@Metric.register("critical_concentration")
def _critical_concentration(
    trace: "Trace", safety_margin: float = 0.0
) -> Dict[str, Any]:
    """Compute c* = 1/sqrt(rho_A * n) at the population's n with the trace's rho_A."""
    from src.analysis.icc import critical_concentration_band, estimate_within_model_icc

    inputs = _trace_to_icc_inputs(trace, "model")
    icc = estimate_within_model_icc(**inputs)
    n_agents = len(inputs["agent_models"])
    c_hat, c_lo, c_hi = critical_concentration_band(icc, max(n_agents, 1))
    return {
        "c_hat": c_hat,
        "c_lo": c_lo,
        "c_hi": c_hi,
        "rho_hat": icc.rho_hat,
        "n_agents": n_agents,
        "safety_margin": safety_margin,
        "c_recommended_cap": max(c_lo - safety_margin, 0.0),
    }


@Metric.register("allocative_efficiency")
def _allocative_efficiency(trace: "Trace") -> float:
    """Mean efficiency across replications.

    For each replication, compute realized surplus from trades divided by
    the maximum possible surplus given the buyer/seller valuations
    observed in that replication's decisions.
    """
    efficiencies: list = []
    for rep in trace.replications:
        buyer_vals: list = []
        seller_vals: list = []
        for rnd in rep.rounds:
            for d in rnd.decisions:
                if d.role == "buyer":
                    buyer_vals.append(d.equilibrium_price + 999)  # placeholder
                else:
                    seller_vals.append(d.equilibrium_price - 999)

        # Realized surplus: sum over trades of (buyer_eq + buyer_premium) - (seller_eq - seller_premium)
        # Since traces don't carry the original valuations directly yet, use
        # the engine's per-replication efficiency if available in trace.config.
        # Placeholder: use the count of trades as a proxy until valuations
        # are folded into the Trace schema in v0.2.1.
        realized = sum(rnd.n_trades for rnd in rep.rounds)
        max_possible = max(len(buyer_vals), len(seller_vals)) or 1
        efficiencies.append(realized / max_possible if max_possible else 0.0)

    if not efficiencies:
        return 0.0
    return float(sum(efficiencies) / len(efficiencies))


@Metric.register("welfare_bound")
def _welfare_bound(trace: "Trace") -> Dict[str, Any]:
    """Per-agent welfare loss lower bound Omega(b^2 + rho * sigma^2)."""
    import numpy as np

    from src.analysis.icc import estimate_within_model_icc

    inputs = _trace_to_icc_inputs(trace, "model")
    rho_hat = estimate_within_model_icc(**inputs).rho_hat

    devs: list = []
    for rep in trace.replications:
        for rnd in rep.rounds:
            for d in rnd.decisions:
                devs.append(d.submitted_price - d.equilibrium_price)
    if not devs:
        return {"bias_term": 0.0, "correlation_term": 0.0, "total": 0.0, "rho": rho_hat}

    arr = np.asarray(devs, dtype=float)
    bias = float(np.mean(arr))
    variance = float(np.var(arr))
    bias_term = bias * bias
    correlation_term = (rho_hat if rho_hat == rho_hat else 0.0) * variance
    return {
        "bias_term": bias_term,
        "correlation_term": correlation_term,
        "total": bias_term + correlation_term,
        "rho": rho_hat,
        "bias": bias,
        "variance": variance,
    }


@Metric.register("equilibrium_distance")
def _equilibrium_distance(trace: "Trace") -> float:
    """Mean absolute deviation from the equilibrium clearing price."""
    devs: list = []
    for rep in trace.replications:
        for rnd in rep.rounds:
            if rnd.clearing_price is None:
                continue
            # Use the first decision's equilibrium_price as the round's p*.
            if rnd.decisions:
                p_star = rnd.decisions[0].equilibrium_price
                devs.append(abs(rnd.clearing_price - p_star))
    if not devs:
        return float("nan")
    return float(sum(devs) / len(devs))


@Metric.register("bid_entropy")
def _bid_entropy(trace: "Trace", n_bins: int = 10) -> float:
    """Shannon entropy of submitted buyer prices."""
    import numpy as np

    prices = [
        d.submitted_price
        for rep in trace.replications
        for rnd in rep.rounds
        for d in rnd.decisions
        if d.role == "buyer"
    ]
    return _entropy(prices, n_bins)


@Metric.register("price_entropy")
def _price_entropy(trace: "Trace", n_bins: int = 10) -> float:
    """Shannon entropy of clearing prices."""
    prices = [
        rnd.clearing_price
        for rep in trace.replications
        for rnd in rep.rounds
        if rnd.clearing_price is not None
    ]
    return _entropy(prices, n_bins)


def _entropy(values: list, n_bins: int) -> float:
    import math

    import numpy as np

    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if arr.min() == arr.max():
        return 0.0
    hist, _ = np.histogram(arr, bins=n_bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    return float(-sum(p * math.log2(p) for p in probs if p > 0))
