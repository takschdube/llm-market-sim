# tests/test_agents.py
"""Tests for agent classes."""
import pytest

from src.agents import AgentState, Order, ZIAgent


class TestAgentState:
    def test_create_buyer(self):
        state = AgentState(
            id="buyer_1",
            endowment={"money": 100, "good_A": 0},
            valuation={"good_A": 20},
            role="buyer"
        )
        assert state.id == "buyer_1"
        assert state.endowment["money"] == 100
        assert state.valuation["good_A"] == 20
        assert state.role == "buyer"

    def test_create_seller(self):
        state = AgentState(
            id="seller_1",
            endowment={"money": 0, "good_A": 10},
            valuation={"good_A": 5},
            role="seller"
        )
        assert state.endowment["good_A"] == 10
        assert state.role == "seller"


class TestOrder:
    def test_create_buy_order(self):
        order = Order(
            agent_id="agent_1",
            side="buy",
            good="good_A",
            price=15.0,
            quantity=1.0
        )
        assert order.side == "buy"
        assert order.price == 15.0

    def test_create_sell_order(self):
        order = Order(
            agent_id="agent_2",
            side="sell",
            good="good_A",
            price=10.0,
            quantity=2.0
        )
        assert order.side == "sell"
        assert order.quantity == 2.0


class TestZIAgent:
    @pytest.fixture
    def buyer_agent(self):
        state = AgentState(
            id="zi_buyer",
            endowment={"money": 100, "good_A": 0},
            valuation={"good_A": 20},
            role="buyer"
        )
        return ZIAgent(state, max_price=30.0)

    @pytest.fixture
    def seller_agent(self):
        state = AgentState(
            id="zi_seller",
            endowment={"money": 0, "good_A": 10},
            valuation={"good_A": 5},
            role="seller"
        )
        return ZIAgent(state, max_price=30.0)

    def test_buyer_makes_valid_order(self, buyer_agent):
        """Buyer should make buy orders with price in [0, valuation]."""
        market_info = {"round": 1, "bids": [], "asks": [], "last_price": None}

        # Run multiple times since it's stochastic
        for _ in range(10):
            buyer_agent.reset_for_new_round()  # Reset traded flag
            order = buyer_agent.decide(market_info)
            assert order is not None
            assert order.side == "buy"
            assert order.good == "good_A"
            # ZI-C constraint: price in [0, valuation]
            assert 0 <= order.price <= buyer_agent.state.valuation["good_A"]

    def test_seller_makes_valid_order(self, seller_agent):
        """Seller should make sell orders with price in [cost, max_price]."""
        market_info = {"round": 1, "bids": [], "asks": [], "last_price": None}

        for _ in range(10):
            seller_agent.reset_for_new_round()  # Reset traded flag
            order = seller_agent.decide(market_info)
            assert order is not None
            assert order.side == "sell"
            assert order.good == "good_A"
            # ZI-C constraint: price in [valuation, max_price]
            assert seller_agent.state.valuation["good_A"] <= order.price <= seller_agent.max_price

    def test_single_trade_per_round(self, buyer_agent):
        """Agent should only trade once per round."""
        market_info = {"round": 1, "bids": [], "asks": [], "last_price": None}

        # First decision should work
        order1 = buyer_agent.decide(market_info)
        assert order1 is not None

        # Mark as traded
        buyer_agent.mark_traded()

        # Second decision should return None
        order2 = buyer_agent.decide(market_info)
        assert order2 is None

        # After reset, should work again
        buyer_agent.reset_for_new_round()
        order3 = buyer_agent.decide(market_info)
        assert order3 is not None

    def test_max_price_configuration(self):
        """Test that max_price is properly configured."""
        state = AgentState(
            id="seller",
            endowment={"money": 0, "good_A": 10},
            valuation={"good_A": 5},
            role="seller"
        )

        # Explicit max_price
        agent1 = ZIAgent(state, max_price=25.0)
        assert agent1.max_price == 25.0

        # Auto-calculated (2x valuation)
        agent2 = ZIAgent(state)
        assert agent2.max_price == 10.0  # 2 * 5

    def test_buyer_no_longer_constrained_by_cash(self, buyer_agent):
        """
        Buyer bids should NOT be constrained by cash endowment.

        This tests the G&S fix: the budget constraint is conceptual
        (can't bid above valuation), not a cash constraint.
        """
        # Set very low cash - should NOT affect bidding
        buyer_agent.state.endowment["money"] = 1
        market_info = {"round": 1, "bids": [], "asks": [], "last_price": None}

        # Agent should still bid up to valuation, not limited by cash
        bids = []
        for _ in range(50):
            buyer_agent.reset_for_new_round()
            order = buyer_agent.decide(market_info)
            if order:
                bids.append(order.price)

        # Should see bids above the cash amount (1) but below valuation (20)
        assert max(bids) > 1, "Bids should not be constrained by cash"
        assert max(bids) <= 20, "Bids should still respect valuation"
