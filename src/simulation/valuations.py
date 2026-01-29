# src/simulation/valuations.py
"""
Valuation Scheme Framework
==========================
Provides pluggable valuation assignment strategies for market experiments.

Based on Vernon Smith's Induced Value Theory:
- Buyers receive "induced values" (reservation prices)
- Sellers receive "induced costs" (minimum acceptable prices)
- Heterogeneous values create realistic supply/demand curves

Available Schemes:
- LinearValuationScheme: Smith-style linear induced values (default)
- UniformValuationScheme: Random values with seed for reproducibility
- SymmetricValuationScheme: Symmetric around equilibrium price
- FixedValuationScheme: Explicit values for exact replication

Example:
    from src.simulation.valuations import LinearValuationScheme, get_scheme

    # Use default linear scheme
    scheme = LinearValuationScheme()
    profiles = scheme.generate_profiles(n_buyers=3, n_sellers=3)

    # Or use factory function
    scheme = get_scheme("uniform", seed=42)
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentProfile:
    """Complete initial state for one agent."""

    role: str  # "buyer" or "seller"
    valuation: float
    endowment: Dict[str, float]


class ValuationScheme(ABC):
    """
    Abstract base for valuation assignment strategies.

    Subclasses implement different approaches to assigning
    valuations and endowments to market participants.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scheme identifier for results serialization."""
        pass

    @abstractmethod
    def generate_profiles(self, n_buyers: int, n_sellers: int) -> List[AgentProfile]:
        """
        Generate agent profiles with valuations and endowments.

        Args:
            n_buyers: Number of buyer agents
            n_sellers: Number of seller agents

        Returns:
            List of AgentProfile objects (buyers first, then sellers)
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Serialize scheme configuration for reproducibility.

        Returns:
            Dict with 'name' and 'params' keys
        """
        pass


class LinearValuationScheme(ValuationScheme):
    """
    Smith-style linear induced values.

    Creates linearly spaced valuations within buyer/seller groups.
    This is the classic experimental economics approach used in
    Gode & Sunder (1993) and many subsequent studies.

    Default configuration (for 6 buyers, 6 sellers):
        Buyers: valuations = [60, 65, 70, 75, 80, 85]
        Sellers: costs = [5, 10, 15, 20, 25, 30]

    This creates a clear price range where trades are beneficial
    with equilibrium price around 45 (midpoint of marginal pair).
    The wide gap between buyer valuations and seller costs ensures
    good bid/ask overlap even with ZI random pricing.
    """

    def __init__(
        self,
        buyer_base: float = 60.0,
        buyer_step: float = 5.0,
        seller_base: float = 5.0,
        seller_step: float = 5.0,
        buyer_money: float = 100.0,
        seller_goods: float = 10.0,
    ):
        """
        Args:
            buyer_base: Starting valuation for first buyer
            buyer_step: Increment between buyer valuations
            seller_base: Starting cost for first seller
            seller_step: Increment between seller costs
            buyer_money: Initial money endowment for buyers
            seller_goods: Initial goods endowment for sellers
        """
        self.buyer_base = buyer_base
        self.buyer_step = buyer_step
        self.seller_base = seller_base
        self.seller_step = seller_step
        self.buyer_money = buyer_money
        self.seller_goods = seller_goods

    @property
    def name(self) -> str:
        return "linear"

    def generate_profiles(self, n_buyers: int, n_sellers: int) -> List[AgentProfile]:
        profiles = []

        # Generate buyers (high valuations, start with money)
        for i in range(n_buyers):
            profiles.append(
                AgentProfile(
                    role="buyer",
                    valuation=self.buyer_base + i * self.buyer_step,
                    endowment={"money": self.buyer_money, "good_A": 0},
                )
            )

        # Generate sellers (low valuations/costs, start with goods)
        for i in range(n_sellers):
            profiles.append(
                AgentProfile(
                    role="seller",
                    valuation=self.seller_base + i * self.seller_step,
                    endowment={"money": 0, "good_A": self.seller_goods},
                )
            )

        return profiles

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": {
                "buyer_base": self.buyer_base,
                "buyer_step": self.buyer_step,
                "seller_base": self.seller_base,
                "seller_step": self.seller_step,
                "buyer_money": self.buyer_money,
                "seller_goods": self.seller_goods,
            },
        }


class UniformValuationScheme(ValuationScheme):
    """
    Uniform random valuations within specified ranges.

    Useful for testing robustness to valuation distributions
    and ensuring results don't depend on specific value patterns.

    IMPORTANT: Always provide a seed for reproducibility in experiments.
    """

    def __init__(
        self,
        buyer_min: float = 15.0,
        buyer_max: float = 25.0,
        seller_min: float = 5.0,
        seller_max: float = 15.0,
        buyer_money: float = 100.0,
        seller_goods: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            buyer_min: Minimum buyer valuation
            buyer_max: Maximum buyer valuation
            seller_min: Minimum seller cost
            seller_max: Maximum seller cost
            buyer_money: Initial money endowment for buyers
            seller_goods: Initial goods endowment for sellers
            seed: Random seed for reproducibility (required for experiments)
        """
        self.buyer_min = buyer_min
        self.buyer_max = buyer_max
        self.seller_min = seller_min
        self.seller_max = seller_max
        self.buyer_money = buyer_money
        self.seller_goods = seller_goods
        self.seed = seed

    @property
    def name(self) -> str:
        return "uniform"

    def generate_profiles(self, n_buyers: int, n_sellers: int) -> List[AgentProfile]:
        # Use dedicated RNG for reproducibility
        rng = random.Random(self.seed)
        profiles = []

        # Generate buyers
        for _ in range(n_buyers):
            profiles.append(
                AgentProfile(
                    role="buyer",
                    valuation=rng.uniform(self.buyer_min, self.buyer_max),
                    endowment={"money": self.buyer_money, "good_A": 0},
                )
            )

        # Generate sellers
        for _ in range(n_sellers):
            profiles.append(
                AgentProfile(
                    role="seller",
                    valuation=rng.uniform(self.seller_min, self.seller_max),
                    endowment={"money": 0, "good_A": self.seller_goods},
                )
            )

        return profiles

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": {
                "buyer_min": self.buyer_min,
                "buyer_max": self.buyer_max,
                "seller_min": self.seller_min,
                "seller_max": self.seller_max,
                "buyer_money": self.buyer_money,
                "seller_goods": self.seller_goods,
                "seed": self.seed,
            },
        }


class SymmetricValuationScheme(ValuationScheme):
    """
    Symmetric valuations around equilibrium price.

    Creates equal gains from trade for both buyers and sellers,
    useful for studying whether agents treat buying/selling symmetrically.

    Buyers: eq_price + spread/2, eq_price + spread/2 - step, ...
    Sellers: eq_price - spread/2, eq_price - spread/2 + step, ...
    """

    def __init__(
        self,
        equilibrium_price: float = 12.5,
        spread: float = 15.0,
        step: float = 2.0,
        buyer_money: float = 100.0,
        seller_goods: float = 10.0,
    ):
        """
        Args:
            equilibrium_price: Target equilibrium price
            spread: Total spread between highest buyer and lowest seller
            step: Increment between valuations within each group
            buyer_money: Initial money endowment for buyers
            seller_goods: Initial goods endowment for sellers
        """
        self.equilibrium_price = equilibrium_price
        self.spread = spread
        self.step = step
        self.buyer_money = buyer_money
        self.seller_goods = seller_goods

    @property
    def name(self) -> str:
        return "symmetric"

    def generate_profiles(self, n_buyers: int, n_sellers: int) -> List[AgentProfile]:
        profiles = []
        half_spread = self.spread / 2

        # Buyers: start high and decrease
        # Highest buyer values at eq_price + half_spread
        for i in range(n_buyers):
            val = self.equilibrium_price + half_spread - i * self.step
            profiles.append(
                AgentProfile(
                    role="buyer",
                    valuation=val,
                    endowment={"money": self.buyer_money, "good_A": 0},
                )
            )

        # Sellers: start low and increase
        # Lowest seller costs at eq_price - half_spread
        for i in range(n_sellers):
            val = self.equilibrium_price - half_spread + i * self.step
            profiles.append(
                AgentProfile(
                    role="seller",
                    valuation=val,
                    endowment={"money": 0, "good_A": self.seller_goods},
                )
            )

        return profiles

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": {
                "equilibrium_price": self.equilibrium_price,
                "spread": self.spread,
                "step": self.step,
                "buyer_money": self.buyer_money,
                "seller_goods": self.seller_goods,
            },
        }


class FixedValuationScheme(ValuationScheme):
    """
    Explicit valuations for full experimental control.

    Allows specifying exact buyer/seller valuations for replication
    of specific experimental conditions from the literature.
    """

    def __init__(
        self,
        buyer_valuations: List[float],
        seller_valuations: List[float],
        buyer_money: float = 100.0,
        seller_goods: float = 10.0,
    ):
        """
        Args:
            buyer_valuations: List of buyer valuations (one per buyer)
            seller_valuations: List of seller costs (one per seller)
            buyer_money: Initial money endowment for buyers
            seller_goods: Initial goods endowment for sellers
        """
        self.buyer_valuations = buyer_valuations
        self.seller_valuations = seller_valuations
        self.buyer_money = buyer_money
        self.seller_goods = seller_goods

    @property
    def name(self) -> str:
        return "fixed"

    def generate_profiles(self, n_buyers: int, n_sellers: int) -> List[AgentProfile]:
        if n_buyers != len(self.buyer_valuations):
            raise ValueError(
                f"Expected {len(self.buyer_valuations)} buyers, got {n_buyers}"
            )
        if n_sellers != len(self.seller_valuations):
            raise ValueError(
                f"Expected {len(self.seller_valuations)} sellers, got {n_sellers}"
            )

        profiles = []

        for val in self.buyer_valuations:
            profiles.append(
                AgentProfile(
                    role="buyer",
                    valuation=val,
                    endowment={"money": self.buyer_money, "good_A": 0},
                )
            )

        for val in self.seller_valuations:
            profiles.append(
                AgentProfile(
                    role="seller",
                    valuation=val,
                    endowment={"money": 0, "good_A": self.seller_goods},
                )
            )

        return profiles

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": {
                "buyer_valuations": self.buyer_valuations,
                "seller_valuations": self.seller_valuations,
                "buyer_money": self.buyer_money,
                "seller_goods": self.seller_goods,
            },
        }


# Registry for CLI access and factory function
VALUATION_SCHEMES = {
    "linear": LinearValuationScheme,
    "uniform": UniformValuationScheme,
    "symmetric": SymmetricValuationScheme,
    "fixed": FixedValuationScheme,
}


def get_scheme(name: str, **kwargs) -> ValuationScheme:
    """
    Factory function for scheme instantiation.

    Args:
        name: Scheme name ('linear', 'uniform', 'symmetric', 'fixed')
        **kwargs: Scheme-specific parameters

    Returns:
        Configured ValuationScheme instance

    Example:
        scheme = get_scheme("linear", buyer_base=25.0)
        scheme = get_scheme("uniform", seed=42)
    """
    if name not in VALUATION_SCHEMES:
        available = list(VALUATION_SCHEMES.keys())
        raise ValueError(f"Unknown scheme: {name}. Available: {available}")
    return VALUATION_SCHEMES[name](**kwargs)
