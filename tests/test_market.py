# tests/test_market.py
"""Tests for market mechanism."""
import pytest

from src.agents import Order
from src.market.mechanism import DoubleAuction, Trade


class TestTrade:
    def test_create_trade(self):
        trade = Trade(
            buyer_id="buyer_1",
            seller_id="seller_1",
            good="good_A",
            price=12.5,
            quantity=1.0
        )
        assert trade.buyer_id == "buyer_1"
        assert trade.seller_id == "seller_1"
        assert trade.price == 12.5


class TestDoubleAuction:
    @pytest.fixture
    def market(self):
        return DoubleAuction()

    def test_submit_bid(self, market):
        order = Order("buyer_1", "buy", "good_A", 15.0, 1.0)
        market.submit(order)
        assert len(market.bids) == 1
        assert len(market.asks) == 0

    def test_submit_ask(self, market):
        order = Order("seller_1", "sell", "good_A", 10.0, 1.0)
        market.submit(order)
        assert len(market.bids) == 0
        assert len(market.asks) == 1

    def test_clear_matching_orders(self, market):
        """When bid >= ask, a trade should occur at midpoint price."""
        bid = Order("buyer_1", "buy", "good_A", 15.0, 1.0)
        ask = Order("seller_1", "sell", "good_A", 10.0, 1.0)

        market.submit(bid)
        market.submit(ask)

        trades = market.clear()

        assert len(trades) == 1
        assert trades[0].buyer_id == "buyer_1"
        assert trades[0].seller_id == "seller_1"
        assert trades[0].price == 12.5  # midpoint of 15 and 10
        assert trades[0].quantity == 1.0

    def test_no_trade_when_bid_less_than_ask(self, market):
        """When bid < ask, no trade should occur."""
        bid = Order("buyer_1", "buy", "good_A", 8.0, 1.0)
        ask = Order("seller_1", "sell", "good_A", 10.0, 1.0)

        market.submit(bid)
        market.submit(ask)

        trades = market.clear()

        assert len(trades) == 0
        # Orders should remain in the book
        assert len(market.bids) == 1
        assert len(market.asks) == 1

    def test_multiple_trades(self, market):
        """Multiple matching pairs should all trade."""
        # Two buyers and two sellers
        market.submit(Order("buyer_1", "buy", "good_A", 20.0, 1.0))
        market.submit(Order("buyer_2", "buy", "good_A", 18.0, 1.0))
        market.submit(Order("seller_1", "sell", "good_A", 10.0, 1.0))
        market.submit(Order("seller_2", "sell", "good_A", 12.0, 1.0))

        trades = market.clear()

        assert len(trades) == 2
        # Highest bid matches with lowest ask first
        assert trades[0].buyer_id == "buyer_1"
        assert trades[0].seller_id == "seller_1"

    def test_partial_clearing(self, market):
        """Only matching orders should clear."""
        market.submit(Order("buyer_1", "buy", "good_A", 20.0, 1.0))
        market.submit(Order("buyer_2", "buy", "good_A", 5.0, 1.0))  # Too low
        market.submit(Order("seller_1", "sell", "good_A", 10.0, 1.0))

        trades = market.clear()

        assert len(trades) == 1
        # buyer_2's order should remain (too low to match)
        assert len(market.bids) == 1
        assert market.bids[0].agent_id == "buyer_2"

    def test_get_market_info_empty(self, market):
        info = market.get_market_info()
        assert info["bids"] == []
        assert info["asks"] == []
        assert info["last_price"] is None

    def test_get_market_info_with_orders(self, market):
        market.submit(Order("buyer_1", "buy", "good_A", 15.0, 1.0))
        market.submit(Order("seller_1", "sell", "good_A", 10.0, 1.0))

        info = market.get_market_info()

        assert len(info["bids"]) == 1
        assert len(info["asks"]) == 1
        assert info["bids"][0] == (15.0, 1.0)  # (price, quantity)

    def test_get_market_info_with_last_trade(self, market):
        market.submit(Order("buyer_1", "buy", "good_A", 15.0, 1.0))
        market.submit(Order("seller_1", "sell", "good_A", 10.0, 1.0))
        market.clear()

        info = market.get_market_info()

        assert info["last_price"] == 12.5
