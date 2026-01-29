# tests/test_metrics.py
"""Tests for analysis metrics."""
import pytest

from src.agents import AgentState, ZIAgent
from src.market.mechanism import Trade
from src.analysis.metrics import (
    compute_efficiency,
    compute_max_surplus,
    compute_convergence,
    compute_walrasian_price,
)


@pytest.fixture
def simple_agents():
    """Create a simple 2-agent economy: 1 buyer, 1 seller."""
    buyer = ZIAgent(AgentState(
        id="buyer",
        endowment={"money": 100, "good_A": 0},
        valuation={"good_A": 20},  # Willing to pay up to 20
        role="buyer"
    ))
    seller = ZIAgent(AgentState(
        id="seller",
        endowment={"money": 0, "good_A": 10},
        valuation={"good_A": 5},  # Willing to sell at 5 or higher
        role="seller"
    ))
    return [buyer, seller]


@pytest.fixture
def multi_agents():
    """Create a 4-agent economy: 2 buyers, 2 sellers with different valuations."""
    agents = [
        ZIAgent(AgentState(
            id="buyer_high",
            endowment={"money": 100, "good_A": 0},
            valuation={"good_A": 25},
            role="buyer"
        )),
        ZIAgent(AgentState(
            id="buyer_low",
            endowment={"money": 100, "good_A": 0},
            valuation={"good_A": 15},
            role="buyer"
        )),
        ZIAgent(AgentState(
            id="seller_low",
            endowment={"money": 0, "good_A": 10},
            valuation={"good_A": 5},
            role="seller"
        )),
        ZIAgent(AgentState(
            id="seller_high",
            endowment={"money": 0, "good_A": 10},
            valuation={"good_A": 10},
            role="seller"
        )),
    ]
    return agents


class TestComputeMaxSurplus:
    def test_simple_case(self, simple_agents):
        """Max surplus = buyer_val - seller_val = 20 - 5 = 15."""
        surplus = compute_max_surplus(simple_agents)
        assert surplus == 15.0

    def test_multi_agent(self, multi_agents):
        """
        Efficient trades:
        1. buyer_high (25) with seller_low (5): surplus = 20
        2. buyer_low (15) with seller_high (10): surplus = 5
        Total = 25
        """
        surplus = compute_max_surplus(multi_agents)
        assert surplus == 25.0

    def test_no_gains_from_trade(self):
        """When buyer valuation < seller valuation, no surplus possible."""
        buyer = ZIAgent(AgentState(
            id="buyer", endowment={"money": 100, "good_A": 0}, valuation={"good_A": 5}, role="buyer"
        ))
        seller = ZIAgent(AgentState(
            id="seller", endowment={"money": 0, "good_A": 10}, valuation={"good_A": 10}, role="seller"
        ))
        surplus = compute_max_surplus([buyer, seller])
        assert surplus == 0.0


class TestComputeEfficiency:
    def test_perfect_efficiency(self, simple_agents):
        """When the correct agents trade, efficiency should be 1.0."""
        # Trade between buyer (val=20) and seller (val=5)
        trades = [Trade("buyer", "seller", "good_A", 12.5, 1.0)]
        efficiency = compute_efficiency(simple_agents, trades)
        assert efficiency == 1.0

    def test_no_trades(self, simple_agents):
        """No trades means 0 efficiency."""
        efficiency = compute_efficiency(simple_agents, [])
        assert efficiency == 0.0

    def test_partial_efficiency(self, multi_agents):
        """Trade that doesn't maximize surplus should be < 1.0."""
        # Trade buyer_low (15) with seller_low (5): surplus = 10
        # But max surplus = 25, so efficiency = 10/25 = 0.4
        trades = [Trade("buyer_low", "seller_low", "good_A", 10.0, 1.0)]
        efficiency = compute_efficiency(multi_agents, trades)
        assert efficiency == pytest.approx(0.4)


class TestComputeConvergence:
    def test_perfect_convergence(self):
        """Prices at equilibrium means 0 deviation."""
        prices = [10.0, 10.0, 10.0]
        eq_price = 10.0
        mad = compute_convergence(prices, eq_price)
        assert mad == 0.0

    def test_some_deviation(self):
        """MAD should be average of absolute deviations."""
        prices = [8.0, 10.0, 12.0]
        eq_price = 10.0
        # Deviations: 2, 0, 2 -> mean = 4/3
        mad = compute_convergence(prices, eq_price)
        assert mad == pytest.approx(4.0 / 3.0)

    def test_empty_prices(self):
        """Empty price list returns infinity."""
        mad = compute_convergence([], 10.0)
        assert mad == float("inf")


class TestComputeWalrasianPrice:
    def test_simple_case(self, simple_agents):
        """Equilibrium price is midpoint of marginal buyer/seller valuations."""
        # Demand: [20], Supply: [5]
        # Intersection at first pair: (20 + 5) / 2 = 12.5
        eq_price = compute_walrasian_price(simple_agents)
        assert eq_price == 12.5

    def test_multi_agent(self, multi_agents):
        """
        Demand (sorted desc): [25, 15]
        Supply (sorted asc): [5, 10]

        Trade 1: 25 >= 5 -> price = 15 (midpoint)
        Trade 2: 15 >= 10 -> price = 12.5 (midpoint)

        Last valid equilibrium price = 12.5
        """
        eq_price = compute_walrasian_price(multi_agents)
        assert eq_price == 12.5

    def test_no_possible_trades(self):
        """When no gains from trade, equilibrium price is 0."""
        buyer = ZIAgent(AgentState(
            id="buyer", endowment={"money": 100, "good_A": 0}, valuation={"good_A": 5}, role="buyer"
        ))
        seller = ZIAgent(AgentState(
            id="seller", endowment={"money": 0, "good_A": 10}, valuation={"good_A": 10}, role="seller"
        ))
        eq_price = compute_walrasian_price([buyer, seller])
        assert eq_price == 0.0
