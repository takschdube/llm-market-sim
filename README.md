# llm-market-sim

A design and governance toolkit for multi-AI-agent markets. The library simulates
call auctions and continuous double auctions populated by LLM agents, classical
Zero-Intelligence traders (Gode and Sunder, 1993), or mixed populations, and
measures the correlation and welfare effects studied in the cognitive monoculture
paper.

The public surface is eight types plus a small `config` namespace. A market spec,
a population spec, and a study compose declaratively; calling `Study.run()`
executes the experiment and returns an audit-schema trace that you can evaluate,
export to Parquet, and check against a deployment policy.

## Install

```bash
git clone https://github.com/takschdube/llm-market-sim.git
cd llm-market-sim
uv sync
```

The Zero-Intelligence baseline runs offline. LLM agents require provider keys:

```bash
cp .env.example .env
# add the keys for the providers you intend to run
```

## Quick start

No API key is needed for this example; it runs on the classical baseline.

```python
from llm_market_sim import Agent, Market, Metric, Population, Strategy, Study
from llm_market_sim.config import Seed

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
results = trace.evaluate([Metric.allocative_efficiency(), Metric.icc_within()])
print(results)
```

The runnable version is `examples/quickstart_zi.py`. The Zero-Intelligence
population returns a within-model correlation near zero, which is the
uncorrelated control the paper compares LLM populations against.

## The eight types

| Type | Role |
|------|------|
| `Market` | Builds a `MarketSpec`: mechanism, valuations, clearing rule, fees |
| `Strategy` | Decision rule: `zi()`, `llm_prompt(...)`, or `custom(...)` |
| `Agent` | Binds a strategy to a backend: `classical(...)` or `llm(...)` |
| `Population` | An ordered, possibly heterogeneous set of agent specs |
| `Study` | A market, a population, replications, seed, budget; `.run()` returns a `Trace` |
| `Trace` | The audit-schema record of a run; `.evaluate(...)`, `.to_parquet(...)` |
| `Metric` | Named measurements: `icc_within()`, `allocative_efficiency()`, `critical_concentration()` |
| `Governance` | Fleet diversity, audit log, and deployment-policy checks over a trace |

## Adding LLM agents

Mix providers to study cross-architecture correlation. Concentration is the LLM
share of the population.

```python
from llm_market_sim import Agent, Population, Strategy

population = Population.of(
    Agent.llm(
        provider="deepseek",
        strategy=Strategy.llm_prompt(reasoning="react", prompt_version="v1"),
        lineage_tag="deepseek/default",
        count=6,
    ),
    Agent.classical(strategy=Strategy.zi(), count=6),
)
```

A `Study.budget` caps API spend and halts before it is exceeded:

```python
from llm_market_sim.config import Budget, Seed

study = Study(
    name="cm_mixed",
    market=market,
    population=population,
    replications=30,
    seed=Seed(base=42),
    budget=Budget(usd=15, on_exceed="halt"),
)
trace = study.run(parallel=8)
```

## Metrics and governance

```python
from llm_market_sim import Governance, Metric, Trace

trace = Trace.from_parquet("runs/cm_mixed.parquet")

results = trace.evaluate([
    Metric.icc_within(),               # pairwise within-model correlation rho_A
    Metric.critical_concentration(),   # estimated threshold c*
    Metric.allocative_efficiency(),
])

audit = Governance.audit(trace)
policy = Governance.policy(concentration_limit=0.30, rho_a_max=0.70)
report = policy.check(audit)
print(report.to_markdown())
```

## Reproducing the paper experiments

`Study(memoryless=True)` clears agent history at the start of every round and
suppresses prior-round transcripts from the prompt. It isolates the shared-policy
mechanism from within-episode adaptation. The experiment scripts for the paper
live in `experiments/`, and the offline validation of the ZI baseline against
Gode and Sunder (1993) is:

```bash
uv run python experiments/exp00_zi_validation.py
```

## Command line

For a single run without writing Python:

```bash
uv run python main.py --agent-type zi --agents 4 --rounds 10
uv run python main.py --agent-type react --agents 4 --rounds 10 --provider deepseek
```

## Tests

```bash
uv run --extra dev pytest tests/ -q
```

## Citation

See `CITATION.cff`. The accompanying paper is "Cognitive Monoculture in
LLM-Populated Markets: A Critical Concentration Threshold".

```bibtex
@software{llmmarketsim,
  author  = {Dube, Taksch},
  title   = {{llm-market-sim}: A Research Library for Simulating Markets Populated by {LLM} and Zero-Intelligence Agents},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/takschdube/llm-market-sim}
}
```

## License

Apache License 2.0. See `LICENSE`.
