# src/market/mechanism.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.agents import Order


@dataclass
class Trade:
    buyer_id: str
    seller_id: str
    good: str
    price: float
    quantity: float


class DoubleAuction:
    def __init__(self):
        self.bids: List[Order] = []  # Buy orders
        self.asks: List[Order] = []  # Sell orders
        self.trades: List[Trade] = []

    def submit(self, order: Order):
        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

    def clear(self) -> List[Trade]:
        """Match bids and asks, execute trades."""
        # Sort: highest bids first, lowest asks first
        self.bids.sort(key=lambda x: -x.price)
        self.asks.sort(key=lambda x: x.price)
        
        new_trades = []
        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            
            if best_bid.price >= best_ask.price:
                # Trade at midpoint
                trade_price = (best_bid.price + best_ask.price) / 2
                trade = Trade(
                    buyer_id=best_bid.agent_id,
                    seller_id=best_ask.agent_id,
                    good=best_bid.good,
                    price=trade_price,
                    quantity=min(best_bid.quantity, best_ask.quantity)
                )
                new_trades.append(trade)
                self.trades.append(trade)
                
                # Remove matched orders
                self.bids.pop(0)
                self.asks.pop(0)
            else:
                break  # No more matches possible
        
        return new_trades
    
    def get_market_info(self) -> dict:
        return {
            "bids": [(o.price, o.quantity) for o in self.bids[:5]],
            "asks": [(o.price, o.quantity) for o in self.asks[:5]],
            "last_price": self.trades[-1].price if self.trades else None
        }

    def get_bid_ask_spread(self) -> float | None:
        """Return the bid-ask spread (best_ask - best_bid), or None if no orders."""
        if not self.bids or not self.asks:
            return None
        # Sort to get best prices
        best_bid = max(o.price for o in self.bids)
        best_ask = min(o.price for o in self.asks)
        return best_ask - best_bid