# tests/test_simulation.py
"""Tests for simulation runner."""
import pytest

from src.agents import AgentState, ZIAgent
from src.simulation.runner import Simulation, create_agents_from_scheme


class TestCreateAgents:
    def test_creates_correct_number(self):
        agents = create_agents_from_scheme(n_agents=6, agent_type="zi")
        assert len(agents) == 6

    def test_creates_buyers_and_sellers(self):
        agents = create_agents_from_scheme(n_agents=4, agent_type="zi")

        buyers = [a for a in agents if a.state.endowment.get("money", 0) > 0]
        sellers = [a for a in agents if a.state.endowment.get("good_A", 0) > 0]

        assert len(buyers) == 2
        assert len(sellers) == 2

    def test_buyers_have_higher_valuations(self):
        """Buyers should value goods more than sellers for gains from trade."""
        agents = create_agents_from_scheme(n_agents=4, agent_type="zi")

        buyers = [a for a in agents if a.state.endowment.get("money", 0) > 0]
        sellers = [a for a in agents if a.state.endowment.get("good_A", 0) > 0]

        min_buyer_val = min(a.state.valuation["good_A"] for a in buyers)
        max_seller_val = max(a.state.valuation["good_A"] for a in sellers)

        # There should be potential gains from trade
        assert min_buyer_val > max_seller_val


class TestSimulation:
    @pytest.fixture
    def simple_simulation(self):
        """Create a simple simulation with ZI agents."""
        agents = create_agents_from_scheme(n_agents=4, agent_type="zi")
        return Simulation(agents, num_rounds=5)

    def test_runs_correct_rounds(self, simple_simulation):
        results = simple_simulation.run()
        assert len(results) == 5

    def test_results_have_required_keys(self, simple_simulation):
        results = simple_simulation.run()

        for rd in results:
            assert "round" in rd
            assert "num_trades" in rd
            assert "prices" in rd
            assert "volume" in rd

    def test_round_numbers_are_sequential(self, simple_simulation):
        results = simple_simulation.run()

        round_nums = [rd["round"] for rd in results]
        assert round_nums == [1, 2, 3, 4, 5]

    def test_prices_are_positive(self, simple_simulation):
        results = simple_simulation.run()

        all_prices = [p for rd in results for p in rd["prices"]]
        for price in all_prices:
            assert price > 0

    def test_trades_update_endowments(self):
        """After trades, agent endowments should change."""
        # Create agents with known endowments
        buyer = ZIAgent(AgentState(
            id="buyer",
            endowment={"money": 100.0, "good_A": 0.0},
            valuation={"good_A": 50},  # Very high valuation to ensure trade
            role="buyer"
        ))
        seller = ZIAgent(AgentState(
            id="seller",
            endowment={"money": 0.0, "good_A": 10.0},
            valuation={"good_A": 1},  # Very low valuation to ensure trade
            role="seller"
        ))

        initial_buyer_money = buyer.state.endowment["money"]
        initial_seller_goods = seller.state.endowment["good_A"]

        # Disable replenishment so trades affect endowments permanently
        sim = Simulation([buyer, seller], num_rounds=10, replenish_endowments=False)
        results = sim.run()

        total_trades = sum(rd["num_trades"] for rd in results)

        if total_trades > 0:
            # Buyer should have less money and more goods
            assert buyer.state.endowment["money"] < initial_buyer_money
            assert buyer.state.endowment["good_A"] > 0

            # Seller should have more money and fewer goods
            assert seller.state.endowment["money"] > 0
            assert seller.state.endowment["good_A"] < initial_seller_goods
