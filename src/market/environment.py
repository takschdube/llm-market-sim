# src/market/environment.py
"""
Market Environment
==================
External factors that can affect market dynamics. Use this module
to introduce shocks, information asymmetries, or changing conditions.

This enables research on:
- How LLM agents respond to supply/demand shocks
- Information asymmetry effects
- Market volatility and stability
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random


@dataclass
class MarketEvent:
    """An event that affects market conditions."""
    round: int
    event_type: str
    description: str
    parameters: Dict = field(default_factory=dict)


class Environment(ABC):
    """
    Abstract base class for market environments.

    Environments can modify agent valuations, inject information,
    or change market conditions over time.
    """

    @abstractmethod
    def get_state(self, round_num: int) -> Dict:
        """Get current environment state for a given round."""
        pass

    @abstractmethod
    def apply_effects(self, agents: List, round_num: int) -> List[MarketEvent]:
        """
        Apply environmental effects to agents.

        Returns a list of events that occurred this round.
        """
        pass


class StaticEnvironment(Environment):
    """
    Static environment with no changes.

    This is the default - valuations and conditions remain constant.
    """

    def get_state(self, round_num: int) -> Dict:
        return {"type": "static", "round": round_num}

    def apply_effects(self, agents: List, round_num: int) -> List[MarketEvent]:  # noqa: ARG002
        return []


class ShockEnvironment(Environment):
    """
    Environment that introduces supply or demand shocks.

    Useful for studying how agents adapt to sudden changes.
    """

    def __init__(
        self,
        shock_rounds: Optional[List[int]] = None,
        shock_magnitude: float = 0.2,
        shock_type: str = "demand",  # "demand" or "supply"
    ):
        self.shock_rounds = shock_rounds or []
        self.shock_magnitude = shock_magnitude
        self.shock_type = shock_type
        self.applied_shocks: List[MarketEvent] = []

    def get_state(self, round_num: int) -> Dict:
        is_shock_round = round_num in self.shock_rounds
        return {
            "type": "shock",
            "round": round_num,
            "shock_active": is_shock_round,
            "shock_type": self.shock_type if is_shock_round else None,
        }

    def apply_effects(self, agents: List, round_num: int) -> List[MarketEvent]:
        events = []

        if round_num in self.shock_rounds:
            event = MarketEvent(
                round=round_num,
                event_type=f"{self.shock_type}_shock",
                description=f"{self.shock_type.capitalize()} shock of {self.shock_magnitude:.0%}",
                parameters={"magnitude": self.shock_magnitude},
            )
            events.append(event)
            self.applied_shocks.append(event)

            # Apply the shock to agent valuations
            for agent in agents:
                if self.shock_type == "demand":
                    # Increase buyer valuations
                    if agent.state.endowment.get("money", 0) > 0:
                        for good in agent.state.valuation:
                            agent.state.valuation[good] *= (1 + self.shock_magnitude)
                else:  # supply shock
                    # Decrease seller valuations (they value goods more)
                    if agent.state.endowment.get("good_A", 0) > 0:
                        for good in agent.state.valuation:
                            agent.state.valuation[good] *= (1 + self.shock_magnitude)

        return events


class VolatileEnvironment(Environment):
    """
    Environment with random valuation fluctuations each round.

    Models real-world uncertainty and tests agent robustness.
    """

    def __init__(self, volatility: float = 0.05, seed: Optional[int] = None):
        self.volatility = volatility
        self.rng = random.Random(seed)
        self.history: List[MarketEvent] = []

    def get_state(self, round_num: int) -> Dict:
        return {
            "type": "volatile",
            "round": round_num,
            "volatility": self.volatility,
        }

    def apply_effects(self, agents: List, round_num: int) -> List[MarketEvent]:
        events = []

        for agent in agents:
            for good in agent.state.valuation:
                change = self.rng.gauss(0, self.volatility)
                old_val = agent.state.valuation[good]
                new_val = old_val * (1 + change)
                agent.state.valuation[good] = max(0.01, new_val)  # Floor at 0.01

        event = MarketEvent(
            round=round_num,
            event_type="volatility",
            description=f"Random valuation fluctuation (volatility={self.volatility})",
            parameters={"volatility": self.volatility},
        )
        events.append(event)
        self.history.append(event)

        return events


class InformationEnvironment(Environment):
    """
    Environment with asymmetric information.

    Some agents receive signals about future conditions.
    Useful for studying information effects on market efficiency.
    """

    def __init__(
        self,
        informed_fraction: float = 0.5,
        signal_accuracy: float = 0.8,
        seed: Optional[int] = None,
    ):
        self.informed_fraction = informed_fraction
        self.signal_accuracy = signal_accuracy
        self.rng = random.Random(seed)
        self.informed_agents: List[str] = []
        self.current_signal: Optional[str] = None

    def get_state(self, round_num: int) -> Dict:
        return {
            "type": "information_asymmetry",
            "round": round_num,
            "informed_fraction": self.informed_fraction,
            "signal_accuracy": self.signal_accuracy,
        }

    def apply_effects(self, agents: List, round_num: int) -> List[MarketEvent]:
        events = []

        # Select informed agents (first time only)
        if not self.informed_agents:
            n_informed = int(len(agents) * self.informed_fraction)
            informed = self.rng.sample(agents, n_informed)
            self.informed_agents = [a.state.id for a in informed]

        # Generate a signal (true or false with some accuracy)
        true_direction = self.rng.choice(["up", "down"])
        if self.rng.random() < self.signal_accuracy:
            self.current_signal = true_direction
        else:
            self.current_signal = "down" if true_direction == "up" else "up"

        event = MarketEvent(
            round=round_num,
            event_type="information_signal",
            description=f"Signal sent to {len(self.informed_agents)} agents",
            parameters={
                "signal": self.current_signal,
                "informed_agents": self.informed_agents,
            },
        )
        events.append(event)

        return events
