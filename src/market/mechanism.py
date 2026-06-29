# src/market/mechanism.py
"""Market mechanisms.

A Mechanism collects orders during a round and produces a list of Trades when
cleared. Concrete mechanisms differ in the clearing-price rule.

Two mechanisms are provided.

PairwiseMidpointAuction clears each matched buyer-seller pair at its own
midpoint. This is the variant used in the cognitive-monoculture experiments
to date and is one instance of the Lipschitz family covered by Proposition 3
of Dube (2026).

UniformPriceCallAuction matches Definition 1 of Dube (2026): a single uniform
clearing price computed from the marginal executed bid and ask is applied to
all matched pairs. The two mechanisms support the mechanism-invariance
experiment that empirically validates Proposition 3.

The module preserves the legacy alias DoubleAuction == PairwiseMidpointAuction
so existing experiment scripts continue to run unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.agents import Order


@dataclass
class Trade:
    buyer_id: str
    seller_id: str
    good: str
    price: float
    quantity: float


class Mechanism(ABC):
    """Abstract base class for market mechanisms."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.trades: List[Trade] = []

    def submit(self, order: Order) -> None:
        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

    @abstractmethod
    def clear(self) -> List[Trade]:
        """Match submitted orders and return the resulting Trades."""

    def get_market_info(self) -> Dict:
        return {
            "bids": [(o.price, o.quantity) for o in self.bids[:5]],
            "asks": [(o.price, o.quantity) for o in self.asks[:5]],
            "last_price": self.trades[-1].price if self.trades else None,
            "mechanism": self.name,
        }

    def get_bid_ask_spread(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        return min(o.price for o in self.asks) - max(o.price for o in self.bids)


def _sort_book(bids: List[Order], asks: List[Order]) -> Tuple[List[Order], List[Order]]:
    return (
        sorted(bids, key=lambda o: -o.price),
        sorted(asks, key=lambda o: o.price),
    )


def _marginal_k(bids: List[Order], asks: List[Order]) -> int:
    """Largest k such that the k-th highest bid is at least the k-th lowest ask."""
    k_star = 0
    for k in range(min(len(bids), len(asks))):
        if bids[k].price >= asks[k].price:
            k_star = k + 1
        else:
            break
    return k_star


class PairwiseMidpointAuction(Mechanism):
    """Each matched pair clears at its own midpoint.

    Algorithm. Sort buys descending and sells ascending. While the best
    remaining bid is at least the best remaining ask, execute a trade at
    the midpoint of those two prices and consume both orders.
    """

    name = "pairwise_midpoint"

    def clear(self) -> List[Trade]:
        bids, asks = _sort_book(self.bids, self.asks)
        k_star = _marginal_k(bids, asks)
        if k_star == 0:
            return []
        new_trades: List[Trade] = []
        for i in range(k_star):
            price = (bids[i].price + asks[i].price) / 2
            trade = Trade(
                buyer_id=bids[i].agent_id,
                seller_id=asks[i].agent_id,
                good=bids[i].good,
                price=price,
                quantity=min(bids[i].quantity, asks[i].quantity),
            )
            new_trades.append(trade)
            self.trades.append(trade)
        self.bids = bids[k_star:]
        self.asks = asks[k_star:]
        return new_trades


class UniformPriceCallAuction(Mechanism):
    """Single clearing price for all matched pairs (Definition 1 of the paper).

    Algorithm. Sort buys descending and sells ascending. Let k* be the largest
    k such that bid(k) >= ask(k). The clearing price is the midpoint of the
    marginal executed bid and ask: p_clear = (bid(k*) + ask(k*)) / 2. The
    first k* pairs trade at p_clear.
    """

    name = "uniform_price_call"

    def clear(self) -> List[Trade]:
        bids, asks = _sort_book(self.bids, self.asks)
        k_star = _marginal_k(bids, asks)
        if k_star == 0:
            return []
        p_clear = (bids[k_star - 1].price + asks[k_star - 1].price) / 2
        new_trades: List[Trade] = []
        for i in range(k_star):
            trade = Trade(
                buyer_id=bids[i].agent_id,
                seller_id=asks[i].agent_id,
                good=bids[i].good,
                price=p_clear,
                quantity=min(bids[i].quantity, asks[i].quantity),
            )
            new_trades.append(trade)
            self.trades.append(trade)
        self.bids = bids[k_star:]
        self.asks = asks[k_star:]
        return new_trades


# Legacy alias preserved for the experiment scripts.
DoubleAuction = PairwiseMidpointAuction


MECHANISMS: Dict[str, type] = {
    "pairwise_midpoint": PairwiseMidpointAuction,
    "uniform_price_call": UniformPriceCallAuction,
    "double_auction": PairwiseMidpointAuction,
}


def get_mechanism(name: str) -> Mechanism:
    """Factory for mechanism instances."""
    if name not in MECHANISMS:
        raise ValueError(
            f"Unknown mechanism: {name!r}. Choose from {sorted(MECHANISMS.keys())}"
        )
    return MECHANISMS[name]()


def list_mechanisms() -> List[Dict[str, str]]:
    return [
        {"name": cls.name, "class": cls.__name__}
        for name, cls in MECHANISMS.items()
        if name == cls.name
    ]
