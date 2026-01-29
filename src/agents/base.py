# src/agents/base.py
"""
Core agent abstractions for market simulation.

This module defines the fundamental building blocks:
- AgentState: Agent identity, holdings, and role
- Order: Trading order representation
- DecisionLog: Full reasoning trace for research auditability
- BaseAgent: Abstract base class all agents inherit from
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class AgentState:
    """
    Agent identity and state.

    Attributes:
        id: Unique agent identifier
        endowment: Current holdings (e.g., {"good_A": 10, "money": 100})
        valuation: Willingness to pay (e.g., {"good_A": 15})
        role: Fixed at creation as "buyer" or "seller"
    """
    id: str
    endowment: Dict[str, float]
    valuation: Dict[str, float]
    role: Literal["buyer", "seller"]


@dataclass
class Order:
    """
    Trading order submitted to the market.

    Attributes:
        agent_id: Who is placing the order
        side: "buy" or "sell"
        good: Good being traded
        price: Price per unit
        quantity: Units requested
    """
    agent_id: str
    side: Literal["buy", "sell"]
    good: str
    price: float
    quantity: float


@dataclass
class DecisionLog:
    """
    Full reasoning trace for research auditability.

    Every decision made by an agent is logged here with complete context,
    enabling analysis of agent behavior for datasets and papers.

    Attributes:
        round: Trading round number
        agent_id: Which agent made this decision
        agent_type: Type of agent ("zi", "react", "cot", etc.)
        provider: LLM provider if applicable ("deepseek", "anthropic", etc.)
        model: Model name if applicable
        market_state: Market info at decision time
        observation: What the agent observed
        analysis: Market analysis (CoT mode only)
        reasoning: Strategic reasoning (CoT mode only)
        decision: Parsed decision dict
        raw_response: Full LLM response text
        order: Resulting Order object if any
        error: Error message if decision failed
        tokens_used: LLM tokens consumed
        latency_ms: Decision latency
    """
    round: int
    agent_id: str
    agent_type: str
    provider: Optional[str] = None
    model: Optional[str] = None
    market_state: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    analysis: str = ""
    reasoning: str = ""
    decision: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    order: Optional[Order] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round": self.round,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "provider": self.provider,
            "model": self.model,
            "observation": self.observation,
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "decision": self.decision,
            "order": {
                "side": self.order.side,
                "price": self.order.price,
                "quantity": self.order.quantity,
            } if self.order else None,
            "error": self.error,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    All agents must implement the decide() method. The update() method
    handles post-trade state updates and is shared across all agents.

    Attributes:
        state: Agent's current state (identity, holdings, role)
        history: List of past trade results
        decision_logs: Full reasoning traces for research
    """

    def __init__(self, state: AgentState):
        self.state = state
        self.history: List[Dict] = []
        self.decision_logs: List[DecisionLog] = []

    @abstractmethod
    def decide(self, market_info: Dict) -> Optional[Order]:
        """
        Make a trading decision given current market state.

        Args:
            market_info: Dict containing market state (round, bids, asks, last_price)

        Returns:
            Order to submit, or None to hold
        """
        pass

    def update(self, trade_result: Dict) -> None:
        """
        Update internal state after a trade.

        Args:
            trade_result: Dict with trade info (trade object, round, etc.)
        """
        self.history.append(trade_result)

        trade = trade_result.get("trade")
        if trade is None:
            return

        if self.state.id == trade.buyer_id:
            # Buyer: loses money, gains goods
            self.state.endowment["money"] = self.state.endowment.get("money", 0) - trade.price * trade.quantity
            self.state.endowment[trade.good] = self.state.endowment.get(trade.good, 0) + trade.quantity
        elif self.state.id == trade.seller_id:
            # Seller: gains money, loses goods
            self.state.endowment["money"] = self.state.endowment.get("money", 0) + trade.price * trade.quantity
            self.state.endowment[trade.good] = self.state.endowment.get(trade.good, 0) - trade.quantity

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Export decision logs for research analysis.

        Returns:
            List of dicts that can be serialized to JSON.
        """
        return [log.to_dict() for log in self.decision_logs]
