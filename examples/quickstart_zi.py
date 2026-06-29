"""Quickstart: a Zero-Intelligence call auction, no API key required.

Runs entirely on the classical baseline, so it executes offline and is the
fastest way to confirm an install and see the eight-type SDK surface. To add
LLM agents, set provider keys in `.env` and swap a `Population` entry for
`Agent.llm(provider="deepseek", strategy=Strategy.llm_prompt(), count=...)`.

    uv run python examples/quickstart_zi.py
"""
from llm_market_sim import Agent, Market, Metric, Population, Strategy, Study
from llm_market_sim.config import Seed


def main() -> None:
    market = Market.call_auction(
        n_agents=12,
        rounds=20,
        valuations=Market.valuations.linear(n_each=6),
        clearing=Market.clearing.uniform_price(),
    )

    population = Population.of(
        Agent.classical(strategy=Strategy.zi(), count=12),
    )

    study = Study(
        name="quickstart_zi",
        market=market,
        population=population,
        replications=5,
        seed=Seed(base=42),
    )

    trace = study.run()
    results = trace.evaluate(
        [
            Metric.allocative_efficiency(),
            Metric.icc_within(),
        ]
    )

    print(f"replications: {study.replications}")
    print(f"LLM concentration: {population.llm_concentration():.2f}")
    for name, value in results.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
