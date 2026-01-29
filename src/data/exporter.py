# src/data/exporter.py
"""
Dataset exporter for Kaggle and HuggingFace.

Provides tools to export simulation results to standard formats
for publication and sharing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.runner import Simulation


class DatasetExporter:
    """
    Export simulation results to Kaggle/HuggingFace formats.

    Collects data from multiple simulation trials and exports to:
    - Parquet (efficient, typed)
    - CSV (universal)
    - HuggingFace Hub (direct upload)

    Example:
        exporter = DatasetExporter("cognitive_monoculture_v1")

        for trial_id, sim in enumerate(simulations):
            exporter.add_trial(sim, trial_id)

        exporter.to_parquet("output/")
        exporter.to_huggingface("takschdube/llm-trading-decisions")
    """

    def __init__(self, experiment_name: str, equilibrium_price: Optional[float] = None):
        """
        Initialize the exporter.

        Args:
            experiment_name: Name for this experiment/dataset
            equilibrium_price: Theoretical equilibrium price (for deviation calculations)
        """
        self.experiment_name = experiment_name
        self.equilibrium_price = equilibrium_price
        self.created_at = datetime.now().isoformat()

        # Collected data
        self.decisions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.rounds: List[Dict[str, Any]] = []

    def add_trial(
        self,
        simulation: "Simulation",
        trial_id: int,
        agent_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Extract data from a completed simulation.

        Args:
            simulation: Completed Simulation object
            trial_id: Trial number within experiment
            agent_types: Optional mapping of agent_id -> agent_type
        """
        # Get equilibrium price from simulation if not set
        eq_price = self.equilibrium_price or getattr(simulation, "equilibrium_price", None)

        # Extract decision logs from agents
        self._extract_decisions(simulation, trial_id, agent_types, eq_price)

        # Extract trades
        self._extract_trades(simulation, trial_id, agent_types, eq_price)

        # Extract round aggregates
        self._extract_rounds(simulation, trial_id, eq_price)

    def _extract_decisions(
        self,
        simulation: "Simulation",
        trial_id: int,
        agent_types: Optional[Dict[str, str]],
        eq_price: Optional[float],
    ) -> None:
        """Extract decision-level data from simulation."""
        for agent in simulation.agents:
            agent_type = "unknown"
            provider = None
            model = None

            # Determine agent type
            if agent_types and agent.state.id in agent_types:
                agent_type = agent_types[agent.state.id]
            elif hasattr(agent, "agent_type"):
                agent_type = agent.agent_type
            else:
                # Infer from class name
                class_name = agent.__class__.__name__.lower()
                if "zi" in class_name:
                    agent_type = "zi"
                elif "react" in class_name:
                    agent_type = "react"
                elif "cot" in class_name:
                    agent_type = "cot"

            # Get provider/model for LLM agents
            if hasattr(agent, "provider"):
                provider = agent.provider
            if hasattr(agent, "llm") and hasattr(agent.llm, "model"):
                model = agent.llm.model

            # Extract from decision logs
            for log in agent.decision_logs:
                decision = {
                    "experiment_id": self.experiment_name,
                    "trial_id": trial_id,
                    "round": log.round,
                    "agent_id": log.agent_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "model": model,
                    "role": agent.state.role,
                    "valuation": agent.state.valuation.get("good_A", 0),
                    "action": log.decision.get("action", "unknown"),
                    "price": log.decision.get("price", 0),
                    "quantity": log.decision.get("quantity", 0),
                    "observation": log.observation,
                    "analysis": log.analysis,
                    "reasoning": log.reasoning,
                    "raw_response": log.raw_response,
                    "order_executed": log.order is not None,
                    "equilibrium_price": eq_price,
                    "price_deviation": (
                        log.decision.get("price", 0) - eq_price
                        if eq_price else None
                    ),
                }
                self.decisions.append(decision)

    def _extract_trades(
        self,
        simulation: "Simulation",
        trial_id: int,
        agent_types: Optional[Dict[str, str]],
        eq_price: Optional[float],
    ) -> None:
        """Extract trade-level data from simulation."""
        # Get agent valuations for surplus calculation
        agent_vals = {a.state.id: a.state.valuation.get("good_A", 0) for a in simulation.agents}

        trade_id = 0
        for round_data in simulation.round_results:
            round_num = round_data.get("round", 0)
            for trade in round_data.get("trades", []):
                buyer_val = agent_vals.get(trade.buyer_id, 0)
                seller_val = agent_vals.get(trade.seller_id, 0)
                buyer_surplus = buyer_val - trade.price
                seller_surplus = trade.price - seller_val

                trade_record = {
                    "experiment_id": self.experiment_name,
                    "trial_id": trial_id,
                    "round": round_num,
                    "trade_id": trade_id,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "buyer_valuation": buyer_val,
                    "seller_valuation": seller_val,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "buyer_surplus": buyer_surplus,
                    "seller_surplus": seller_surplus,
                    "total_surplus": buyer_surplus + seller_surplus,
                    "equilibrium_price": eq_price,
                    "price_deviation": trade.price - eq_price if eq_price else None,
                }
                self.trades.append(trade_record)
                trade_id += 1

    def _extract_rounds(
        self,
        simulation: "Simulation",
        trial_id: int,
        eq_price: Optional[float],
    ) -> None:
        """Extract round-level aggregates from simulation."""
        import statistics

        for round_data in simulation.round_results:
            round_num = round_data.get("round", 0)
            trades = round_data.get("trades", [])
            prices = [t.price for t in trades]

            round_record = {
                "experiment_id": self.experiment_name,
                "trial_id": trial_id,
                "round": round_num,
                "n_trades": len(trades),
                "volume": sum(t.quantity for t in trades),
                "avg_price": statistics.mean(prices) if prices else None,
                "min_price": min(prices) if prices else None,
                "max_price": max(prices) if prices else None,
                "price_std": statistics.stdev(prices) if len(prices) > 1 else 0,
                "equilibrium_price": eq_price,
                "mad": (
                    statistics.mean(abs(p - eq_price) for p in prices)
                    if prices and eq_price else None
                ),
            }
            self.rounds.append(round_record)

    def to_parquet(self, output_dir: str | Path) -> Dict[str, Path]:
        """
        Export to Parquet format.

        Args:
            output_dir: Directory to write files

        Returns:
            Dict mapping dataset name to file path
        """
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        if self.decisions:
            df = pd.DataFrame(self.decisions)
            path = output_dir / f"{self.experiment_name}_decisions.parquet"
            df.to_parquet(path, index=False)
            paths["decisions"] = path

        if self.trades:
            df = pd.DataFrame(self.trades)
            path = output_dir / f"{self.experiment_name}_trades.parquet"
            df.to_parquet(path, index=False)
            paths["trades"] = path

        if self.rounds:
            df = pd.DataFrame(self.rounds)
            path = output_dir / f"{self.experiment_name}_rounds.parquet"
            df.to_parquet(path, index=False)
            paths["rounds"] = path

        return paths

    def to_csv(self, output_dir: str | Path) -> Dict[str, Path]:
        """
        Export to CSV format.

        Args:
            output_dir: Directory to write files

        Returns:
            Dict mapping dataset name to file path
        """
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        if self.decisions:
            df = pd.DataFrame(self.decisions)
            path = output_dir / f"{self.experiment_name}_decisions.csv"
            df.to_csv(path, index=False)
            paths["decisions"] = path

        if self.trades:
            df = pd.DataFrame(self.trades)
            path = output_dir / f"{self.experiment_name}_trades.csv"
            df.to_csv(path, index=False)
            paths["trades"] = path

        if self.rounds:
            df = pd.DataFrame(self.rounds)
            path = output_dir / f"{self.experiment_name}_rounds.csv"
            df.to_csv(path, index=False)
            paths["rounds"] = path

        return paths

    def to_huggingface(self, repo_id: str, private: bool = False) -> str:
        """
        Push datasets to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
            private: Whether to make the dataset private

        Returns:
            URL of the uploaded dataset
        """
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi

        datasets = {}

        if self.decisions:
            datasets["decisions"] = Dataset.from_list(self.decisions)

        if self.trades:
            datasets["trades"] = Dataset.from_list(self.trades)

        if self.rounds:
            datasets["rounds"] = Dataset.from_list(self.rounds)

        dataset_dict = DatasetDict(datasets)
        dataset_dict.push_to_hub(repo_id, private=private)

        return f"https://huggingface.co/datasets/{repo_id}"

    def generate_datacard(self) -> str:
        """
        Generate a dataset card (README.md) for HuggingFace.

        Returns:
            Markdown string for the dataset card
        """
        n_decisions = len(self.decisions)
        n_trades = len(self.trades)
        n_rounds = len(self.rounds)
        n_trials = len(set(d["trial_id"] for d in self.decisions)) if self.decisions else 0

        return f"""---
license: apache-2.0
task_categories:
  - tabular-classification
  - text-generation
tags:
  - economics
  - multi-agent
  - llm
  - market-simulation
  - trading
---

# {self.experiment_name}

## Dataset Description

Market simulation data from LLM trading agents in double auction markets.

### Dataset Summary

- **Experiment**: {self.experiment_name}
- **Created**: {self.created_at}
- **Trials**: {n_trials}
- **Total decisions**: {n_decisions:,}
- **Total trades**: {n_trades:,}
- **Total rounds**: {n_rounds:,}

### Supported Tasks

- Analysis of LLM agent trading behavior
- Cognitive monoculture research
- Market microstructure analysis
- Multi-agent coordination studies

## Dataset Structure

### decisions.parquet

One row per agent per round. Contains full reasoning traces for LLM agents.

Key columns:
- `agent_type`: "zi", "react", "cot"
- `provider`, `model`: LLM provider and model
- `observation`, `analysis`, `reasoning`: Chain-of-thought traces
- `action`, `price`: Trading decision
- `price_deviation`: Distance from equilibrium

### trades.parquet

One row per executed trade.

Key columns:
- `buyer_id`, `seller_id`: Trade participants
- `price`, `quantity`: Trade terms
- `buyer_surplus`, `seller_surplus`: Gains from trade

### rounds.parquet

Round-level aggregates.

Key columns:
- `n_trades`, `volume`: Trading activity
- `avg_price`, `price_std`: Price statistics
- `efficiency`, `mad`: Market quality metrics

## Citation

```bibtex
@software{{llm_market_sim,
  title = {{LLM Market Simulation}},
  author = {{Dube, Taksch}},
  year = {{2026}},
  url = {{https://github.com/takschdube/llm-market-sim}}
}}
```

## License

Apache License 2.0
"""

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected data."""
        return {
            "experiment_name": self.experiment_name,
            "n_decisions": len(self.decisions),
            "n_trades": len(self.trades),
            "n_rounds": len(self.rounds),
            "n_trials": len(set(d["trial_id"] for d in self.decisions)) if self.decisions else 0,
        }
