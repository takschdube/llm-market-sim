# src/simulation/config.py
"""
Simulation Configuration
========================
Centralized configuration for experiments. Allows reproducible runs
and easy parameter sweeps for research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


AgentType = Literal["zi", "react", "cot"]


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    agent_type: AgentType = "react"
    model: str = "claude-sonnet-4-5-20250929"
    persona: str = "rational"


@dataclass
class MarketConfig:
    """Configuration for market mechanism."""
    mechanism: Literal["double_auction", "call_market"] = "double_auction"
    clearing_rule: Literal["continuous", "periodic"] = "continuous"
    price_rule: Literal["midpoint", "buyer", "seller"] = "midpoint"


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    Use this to define reproducible experiments with all parameters
    in one place.

    Example:
        config = SimulationConfig(
            n_agents=8,
            n_rounds=50,
            agent_type="cot",
            seed=42
        )
    """
    # Core parameters
    n_agents: int = 4
    n_rounds: int = 10
    agent_type: AgentType = "react"

    # Market configuration
    market: MarketConfig = field(default_factory=MarketConfig)

    # Agent configuration (applies to all agents)
    agent_config: AgentConfig = field(default_factory=AgentConfig)

    # Valuation parameters
    buyer_base_valuation: float = 20.0
    buyer_valuation_step: float = 2.0
    seller_base_valuation: float = 5.0
    seller_valuation_step: float = 2.0
    buyer_endowment_money: float = 100.0
    seller_endowment_goods: float = 10.0

    # Reproducibility
    seed: Optional[int] = None

    # Output
    output_dir: str = "data/results"
    save_decision_logs: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "agent_type": self.agent_type,
            "market": {
                "mechanism": self.market.mechanism,
                "clearing_rule": self.market.clearing_rule,
                "price_rule": self.market.price_rule,
            },
            "valuations": {
                "buyer_base": self.buyer_base_valuation,
                "buyer_step": self.buyer_valuation_step,
                "seller_base": self.seller_base_valuation,
                "seller_step": self.seller_valuation_step,
            },
            "seed": self.seed,
        }


# Pre-defined configurations for common experiments
BASELINE_CONFIG = SimulationConfig(
    n_agents=4,
    n_rounds=10,
    agent_type="zi",
)

LLM_REACT_CONFIG = SimulationConfig(
    n_agents=4,
    n_rounds=10,
    agent_type="react",
)

LLM_COT_CONFIG = SimulationConfig(
    n_agents=4,
    n_rounds=10,
    agent_type="cot",
    save_decision_logs=True,
)

LARGE_MARKET_CONFIG = SimulationConfig(
    n_agents=20,
    n_rounds=50,
    agent_type="react",
)
