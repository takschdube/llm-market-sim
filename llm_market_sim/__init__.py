"""llm-market-sim public API.

The library is a design and governance toolkit for multi-AI-agent markets.
The public surface is eight names plus a small `config` namespace; everything
else under this package is internal implementation that delegates to the
private engine at `src/`.

Quick start::

    from llm_market_sim import (
        Market, Strategy, Agent, Population,
        Study, Trace, Metric, Governance,
    )
    from llm_market_sim.config import Budget, Seed

Typical usage::

    market = Market.call_auction(
        n_agents=12,
        valuations=Market.valuations.linear(n_each=6),
        clearing=Market.clearing.uniform_price(),
        rounds=20,
    )

    population = Population.of(
        Agent.classical(strategy=Strategy.zi(), count=12),
    )

    study = Study(
        name="cm01_zi_baseline",
        market=market,
        population=population,
        replications=5,
        seed=Seed(base=42),
    )

    trace = study.run()
    results = trace.evaluate([
        Metric.icc_within(),
        Metric.allocative_efficiency(),
        Metric.critical_concentration(),
    ])

    audit = Governance.audit(trace)
    policy = Governance.policy(concentration_limit=0.05, rho_a_max=0.7)
    report = policy.check(audit)
    print(report.to_markdown())
"""
from __future__ import annotations

from .markets import Market, MarketSpec, ValuationsSpec, ClearingSpec
from .strategies import Strategy, ZIStrategy, LLMPromptStrategy, CustomStrategy, StrategySpec
from .agents import Agent, AgentSpec
from .populations import Population
from .traces import Trace, Decision, RoundResult
from .studies import Study
from .metrics import Metric, MetricSpec, MetricRegistry
from .governance import (
    Governance,
    AuditLog,
    DeploymentPolicy,
    ComplianceReport,
    FleetDiversity,
)
from .config import Budget, Seed

__version__ = "0.2.0-dev"

__all__ = [
    # The eight public types
    "Market",
    "Strategy",
    "Agent",
    "Population",
    "Study",
    "Trace",
    "Metric",
    "Governance",
    # Companion value types exposed for type-checking
    "MarketSpec",
    "ValuationsSpec",
    "ClearingSpec",
    "ZIStrategy",
    "LLMPromptStrategy",
    "CustomStrategy",
    "StrategySpec",
    "AgentSpec",
    "Decision",
    "RoundResult",
    "MetricSpec",
    "MetricRegistry",
    "AuditLog",
    "DeploymentPolicy",
    "ComplianceReport",
    "FleetDiversity",
    "Budget",
    "Seed",
]
