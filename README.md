# LLM Market Simulation

A modular research framework for studying LLM agent behavior in economic markets. Simulate double-auction trading with LLM agents, Zero-Intelligence baselines, or mixed populations.

## Why This Framework?

- **Plug-and-play LLM agents**: Swap between DeepSeek, Claude, GPT, Gemini with config changes
- **Classical baselines built-in**: Zero-Intelligence (Gode & Sunder 1993) for comparison
- **Full auditability**: Every agent decision logged with reasoning traces
- **Extensible architecture**: Registry pattern for adding new agent types
- **Dataset export**: Export to Parquet/CSV for Kaggle/HuggingFace
- **Reproducible**: Seeded experiments, versioned configs, JSON output

## Quick Start

```bash
# Clone and install
git clone https://github.com/takschdube/llm-market-sim.git
cd llm-market-sim
uv sync

# Set up API key
cp .env.example .env
# Edit .env and add your API keys

# Run a simulation (no API key needed for ZI baseline)
uv run python main.py --agent-type zi --agents 4 --rounds 10

# Run with LLM agents
uv run python main.py --agent-type react --agents 4 --rounds 10
```

## CLI Reference

```
uv run python main.py [OPTIONS]

Options:
  --agents N                    Number of agents (default: 4, split 50/50 buyer/seller)
  --rounds N                    Number of trading rounds (default: 10)
  --agent-type {zi,react,cot}   Agent type (default: react)
  --provider PROVIDER           LLM provider (default: deepseek)
  --model MODEL                 Model name (uses provider default if not set)
  --valuation-scheme SCHEME     Valuation distribution (default: linear)
  --seed N                      Random seed for reproducibility
  --name TEXT                   Optional experiment name
  --output-dir PATH             Results directory (default: data/results)
  --no-plot                     Skip plotting
```

### Agent Types

| Type | LLM Calls | Description |
|------|-----------|-------------|
| `zi` | 0 | Zero-Intelligence baseline - random within budget constraints |
| `react` | 1 | Reactive: observe market state → output decision |
| `cot` | 4 | Chain-of-thought: observe → analyze → reason → decide |

### Valuation Schemes

| Scheme | Description |
|--------|-------------|
| `linear` | Smith-style induced values (default) |
| `uniform` | Random values in range |
| `symmetric` | Symmetric around equilibrium |
| `fixed` | Explicit values (programmatic) |

## Viewing Results

```bash
uv run python -m src.analysis.view_results --list          # List all
uv run python -m src.analysis.view_results                 # Most recent
uv run python -m src.analysis.view_results 20260101001     # Specific run
uv run python -m src.analysis.view_results 20260101001 --plot prices
```

## Project Structure

```
llm-market-sim/
├── main.py                     # CLI entry point
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── base.py                 # BaseAgent, AgentState, Order, DecisionLog
│   │   ├── registry.py             # Agent registration and factory
│   │   ├── zi_agent.py             # Zero-Intelligence baseline
│   │   ├── react_agent.py          # LLM reactive agent
│   │   ├── cot_agent.py            # LLM chain-of-thought agent
│   │   ├── llm_client.py           # Unified LLM client (all providers)
│   │   ├── response_parser.py      # JSON response parsing
│   │   └── prompts.py              # Prompt templates
│   ├── market/                 # Market mechanics
│   │   └── mechanism.py            # Double auction
│   ├── simulation/             # Simulation engine
│   │   ├── runner.py               # Main loop
│   │   ├── config.py               # Configuration
│   │   └── valuations.py           # Valuation schemes
│   ├── analysis/               # Metrics and stats
│   │   ├── metrics.py              # Walrasian price, efficiency, entropy
│   │   ├── statistics.py           # Statistical tests
│   │   └── plotting.py             # Visualization
│   └── data/                   # Dataset export
│       ├── schemas.py              # Column definitions
│       └── exporter.py             # Parquet/CSV/HuggingFace export
├── experiments/                # Experiment scripts
│   └── exp00_zi_validation.py      # ZI validation (no API key)
├── tests/                      # Unit tests (78 tests)
└── notebooks/                  # Jupyter notebooks
```

## LLM Providers

| Provider | Model | Notes |
|----------|-------|-------|
| DeepSeek | `deepseek-chat` | **Default** - fast, cost-efficient |
| DeepSeek | `deepseek-reasoner` | R1 reasoning model |
| Anthropic | `claude-sonnet-4-5-20250929` | Flagship tier |
| Anthropic | `claude-opus-4-20250514` | Highest capability |
| OpenAI | `gpt-5.2` | Flagship tier |
| OpenAI | `gpt-5-mini` | Faster variant |
| Google | `gemini-3-pro` | Flagship tier |
| Google | `gemini-3-flash` | Faster variant |

Configure provider and model via CLI flags or environment variables in `.env`.

## Python API

### Creating Agents

```python
from src.agents import create_agent, AgentState, list_agents

# List available agent types
print(list_agents())  # ['zi', 'react', 'cot']

# Create agent by name
state = AgentState(
    id="agent_0",
    endowment={"money": 100, "good_A": 0},
    valuation={"good_A": 15},
    role="buyer"
)
agent = create_agent("react", state, provider="anthropic")

# Or import directly
from src.agents import ZIAgent, ReactAgent, CoTAgent
zi_agent = ZIAgent(state)
llm_agent = ReactAgent(state, provider="deepseek")
```

### Running Simulations

```python
from src.simulation.runner import Simulation, create_agents_from_scheme
from src.simulation.valuations import LinearValuationScheme

# Create agents with valuation scheme
agents = create_agents_from_scheme(
    n_agents=6,
    agent_type="zi",
    valuation_scheme=LinearValuationScheme()
)

# Run simulation
sim = Simulation(agents, num_rounds=20)
results = sim.run()

# Access decision logs
for agent in agents:
    logs = agent.get_logs()
```

### Adding Custom Agents

```python
from src.agents import register_agent, BaseAgent, AgentState, Order

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def decide(self, market_info: dict) -> Order | None:
        # Your logic here
        return Order(self.state.id, "buy", "good_A", 10.0, 1.0)

# Now available via create_agent("my_agent", state)
```

### Exporting Datasets

```python
from src.data.exporter import DatasetExporter

exporter = DatasetExporter(experiment_id="exp001")
exporter.add_trial(simulation, trial_id=0)
exporter.to_parquet("data/datasets/")
exporter.to_huggingface("username/market-decisions")
```

## Statistical Tools

```python
from src.analysis.statistics import (
    compare_two_conditions,      # Mann-Whitney U with effect sizes
    compare_multiple_conditions, # Kruskal-Wallis with post-hoc
    bootstrap_ci,                # Bootstrap confidence intervals
    cohens_d, cliffs_delta       # Effect size calculations
)
```

## Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
```

## ZI Validation Demo

Validates our Zero-Intelligence implementation against Gode & Sunder (1993). No API key needed:

```bash
uv run python experiments/exp00_zi_validation.py
```

## Background

- **Gode & Sunder (1993)** - Zero-Intelligence traders
- **Vernon Smith (1962)** - Double auction experiments
- **Walrasian equilibrium** - Theoretical price benchmark

## License

Apache License 2.0 - See LICENSE file.

Copyright 2026 Dube International

## Citation

```bibtex
@software{llm_market_sim,
  title = {LLM Market Simulation},
  author = {Dube, Taksch},
  year = {2026},
  url = {https://github.com/takschdube/llm-market-sim}
}
```
