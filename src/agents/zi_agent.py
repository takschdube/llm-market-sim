# src/agents/zi_agent.py
"""
Zero-Intelligence Agent (Gode & Sunder, 1993)
=============================================
ZI-C (Zero-Intelligence Constrained) trader baseline, extended for multi-round markets.

Based on: Gode & Sunder (1993) "Allocative Efficiency of Markets with
Zero-Intelligence Traders", Journal of Political Economy.

Key ZI-C properties (from G&S):
- Buyers submit random bids uniformly in [0, valuation]
- Sellers submit random asks uniformly in [cost, max_price]
- Budget constraint is conceptual: can't bid above value or ask below cost
- No learning, no market observation - purely mechanical

Extensions for multi-round markets:
- Agents trade at most once per round (then sit out)
- Replenished between rounds (reset traded flag)
- max_price is a configurable market parameter

This serves as the control condition demonstrating that market convergence
in LLM agents is not a mechanical artifact of market structure. ZI agents
are memoryless and should NOT show convergence over rounds.
"""
from __future__ import annotations

import random
from typing import Optional

from .base import BaseAgent, AgentState, Order, DecisionLog
from .registry import register_agent


@register_agent("zi")
class ZIAgent(BaseAgent):
    """
    Gode & Sunder (1993) ZI-C baseline, extended for multi-round markets.

    Implements the core ZI-C constraint: random bids/asks within the
    no-loss region. Extended with single-trade-per-round enforcement
    for clean per-agent-per-round observations.

    Args:
        state: AgentState with role, valuation, and endowment
        max_price: Upper bound for seller asks (default: auto-calculated)
                   Should be set such that equilibrium falls within tradeable range

    Example:
        agent = ZIAgent(state, max_price=30.0)
    """

    def __init__(self, state: AgentState, max_price: Optional[float] = None):
        super().__init__(state)

        # max_price for seller asks
        # If not provided, default to 2x the agent's valuation as reasonable upper bound
        if max_price is not None:
            self.max_price = max_price
        else:
            self.max_price = self.state.valuation.get("good_A", 10) * 2

        # Track whether agent has traded this round
        self._traded_this_round = False

    def decide(self, market_info: dict) -> Optional[Order]:
        """
        Generate a random order following ZI-C constraints.

        ZI-C specification (Gode & Sunder 1993):
        - Buyers: bid ~ Uniform[0, valuation]
        - Sellers: ask ~ Uniform[cost, max_price]

        Returns None if agent has already traded this round.
        """
        # Single-trade-per-round enforcement
        if self._traded_this_round:
            return None

        good = "good_A"
        role = self.state.role
        valuation = self.state.valuation[good]

        if role == "buyer":
            # ZI-C buyer: bid uniformly in [0, valuation]
            price = random.uniform(0, valuation)
            order = Order(self.state.id, "buy", good, price, 1.0)
        else:
            # ZI-C seller: ask uniformly in [cost, max_price]
            price = random.uniform(valuation, self.max_price)
            order = Order(self.state.id, "sell", good, price, 1.0)

        # Log the decision
        log = DecisionLog(
            round=market_info.get("round", 0),
            agent_id=self.state.id,
            agent_type="zi",
            provider=None,
            model=None,
            market_state=market_info,
            observation=f"ZI agent, role={role}, valuation={valuation}",
            decision={"action": order.side, "price": order.price, "quantity": order.quantity},
            order=order,
        )
        self.decision_logs.append(log)

        return order

    def mark_traded(self) -> None:
        """Mark that this agent has traded this round."""
        self._traded_this_round = True

    def reset_for_new_round(self) -> None:
        """Reset traded flag for a new round. Called by Simulation."""
        self._traded_this_round = False

    @property
    def traded_this_round(self) -> bool:
        """Whether agent has already traded this round."""
        return self._traded_this_round


# Backward compatibility alias
ZeroIntelligenceAgent = ZIAgent
