# src/data/schemas.py
"""
Dataset schemas for market simulation data.

Defines standard column names and types for exported datasets,
ensuring consistency across experiments and enabling easy analysis.
"""
from __future__ import annotations

from typing import Dict, List, Type

# Decision-level dataset (one row per agent per round)
# This is the richest dataset, containing full reasoning traces
DECISION_COLUMNS: List[str] = [
    # Identifiers
    "experiment_id",      # Unique experiment identifier
    "trial_id",           # Trial number within experiment
    "round",              # Trading round number
    "agent_id",           # Agent identifier

    # Agent metadata
    "agent_type",         # "zi", "react", "cot", etc.
    "provider",           # "deepseek", "anthropic", "openai", "google", or None for ZI
    "model",              # Model name or None for ZI
    "role",               # "buyer" or "seller"
    "valuation",          # Agent's valuation for good_A

    # Decision
    "action",             # "buy", "sell", or "hold"
    "price",              # Submitted price
    "quantity",           # Submitted quantity

    # Reasoning trace (for LLM agents)
    "observation",        # What the agent observed
    "analysis",           # Market analysis (CoT only)
    "reasoning",          # Strategic reasoning (CoT only)
    "raw_response",       # Full LLM response text

    # Outcome
    "order_executed",     # Whether order was submitted
    "trade_occurred",     # Whether a trade resulted

    # Market context
    "equilibrium_price",  # Theoretical equilibrium
    "price_deviation",    # Submitted price - equilibrium
    "last_price",         # Previous transaction price
]

# Decision schema with types for validation
DECISION_SCHEMA: Dict[str, Type] = {
    "experiment_id": str,
    "trial_id": int,
    "round": int,
    "agent_id": str,
    "agent_type": str,
    "provider": str,
    "model": str,
    "role": str,
    "valuation": float,
    "action": str,
    "price": float,
    "quantity": float,
    "observation": str,
    "analysis": str,
    "reasoning": str,
    "raw_response": str,
    "order_executed": bool,
    "trade_occurred": bool,
    "equilibrium_price": float,
    "price_deviation": float,
    "last_price": float,
}


# Trade-level dataset (one row per executed trade)
TRADE_COLUMNS: List[str] = [
    # Identifiers
    "experiment_id",
    "trial_id",
    "round",
    "trade_id",           # Trade number within round

    # Participants
    "buyer_id",
    "seller_id",
    "buyer_type",         # Agent type of buyer
    "seller_type",        # Agent type of seller
    "buyer_valuation",
    "seller_valuation",

    # Trade details
    "price",
    "quantity",

    # Surplus
    "buyer_surplus",      # buyer_valuation - price
    "seller_surplus",     # price - seller_valuation
    "total_surplus",      # buyer_surplus + seller_surplus

    # Market context
    "equilibrium_price",
    "price_deviation",
]

# Trade schema with types
TRADE_SCHEMA: Dict[str, Type] = {
    "experiment_id": str,
    "trial_id": int,
    "round": int,
    "trade_id": int,
    "buyer_id": str,
    "seller_id": str,
    "buyer_type": str,
    "seller_type": str,
    "buyer_valuation": float,
    "seller_valuation": float,
    "price": float,
    "quantity": float,
    "buyer_surplus": float,
    "seller_surplus": float,
    "total_surplus": float,
    "equilibrium_price": float,
    "price_deviation": float,
}


# Round-level aggregates (one row per round)
ROUND_COLUMNS: List[str] = [
    # Identifiers
    "experiment_id",
    "trial_id",
    "round",

    # Trading activity
    "n_trades",           # Number of trades this round
    "volume",             # Total quantity traded

    # Price statistics
    "avg_price",          # Average transaction price
    "min_price",
    "max_price",
    "price_std",          # Standard deviation

    # Market quality
    "efficiency",         # Allocative efficiency (0-1)
    "mad",                # Mean absolute deviation from equilibrium

    # Bid-ask spread
    "best_bid",
    "best_ask",
    "spread",

    # Market context
    "equilibrium_price",
]

# Round schema with types
ROUND_SCHEMA: Dict[str, Type] = {
    "experiment_id": str,
    "trial_id": int,
    "round": int,
    "n_trades": int,
    "volume": float,
    "avg_price": float,
    "min_price": float,
    "max_price": float,
    "price_std": float,
    "efficiency": float,
    "mad": float,
    "best_bid": float,
    "best_ask": float,
    "spread": float,
    "equilibrium_price": float,
}
