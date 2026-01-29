# src/market/clearing.py
"""
Market Clearing Rules
=====================
Different rules for how trades are priced when a bid and ask match.

This module provides pluggable clearing rules that can be used with
different market mechanisms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from .mechanism import Order


class ClearingRule(ABC):
    """Abstract base class for market clearing rules."""

    @abstractmethod
    def compute_price(self, bid: Order, ask: Order) -> float:
        """
        Compute the trade price given a matching bid and ask.

        Args:
            bid: The buy order (bid price >= ask price)
            ask: The sell order

        Returns:
            The price at which the trade executes.
        """
        pass


class MidpointClearing(ClearingRule):
    """
    Trade executes at midpoint between bid and ask.

    This is the default rule and splits surplus equally between
    buyer and seller.
    """

    def compute_price(self, bid: Order, ask: Order) -> float:
        return (bid.price + ask.price) / 2


class BuyerPriceClearing(ClearingRule):
    """
    Trade executes at the bid price.

    All surplus goes to the seller. Used in some posted-offer markets.
    """

    def compute_price(self, bid: Order, ask: Order) -> float:  # noqa: ARG002
        return bid.price


class SellerPriceClearing(ClearingRule):
    """
    Trade executes at the ask price.

    All surplus goes to the buyer. Used in some auction formats.
    """

    def compute_price(self, bid: Order, ask: Order) -> float:  # noqa: ARG002
        return ask.price


class TimeWeightedClearing(ClearingRule):
    """
    Trade executes at price of the order that arrived first.

    Rewards market makers who provide liquidity.
    """

    def compute_price(self, bid: Order, ask: Order) -> float:
        # Assumes orders have timestamp attribute
        # For now, default to midpoint if no timestamps
        if hasattr(bid, 'timestamp') and hasattr(ask, 'timestamp'):
            if bid.timestamp < ask.timestamp:
                return bid.price
            else:
                return ask.price
        return (bid.price + ask.price) / 2


# Factory function for easy rule selection
def get_clearing_rule(rule_type: Literal["midpoint", "buyer", "seller", "time_weighted"]) -> ClearingRule:
    """
    Get a clearing rule by name.

    Args:
        rule_type: One of "midpoint", "buyer", "seller", "time_weighted"

    Returns:
        An instance of the appropriate ClearingRule subclass.
    """
    rules = {
        "midpoint": MidpointClearing,
        "buyer": BuyerPriceClearing,
        "seller": SellerPriceClearing,
        "time_weighted": TimeWeightedClearing,
    }

    if rule_type not in rules:
        raise ValueError(f"Unknown clearing rule: {rule_type}. Choose from {list(rules.keys())}")

    return rules[rule_type]()
