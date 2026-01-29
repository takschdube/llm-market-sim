# src/simulation/runner.py
from __future__ import annotations

import asyncio
from typing import List, Optional

from src.agents import (
    BaseAgent,
    AgentState,
    create_agent,
    ZIAgent,
    ReactAgent,
    CoTAgent,
)
from src.market.mechanism import DoubleAuction
from src.simulation.valuations import ValuationScheme, LinearValuationScheme


class Simulation:
    def __init__(self, agents: List[BaseAgent], num_rounds: int = 20, parallel: bool = True,
                 replenish_endowments: bool = True):
        """
        Initialize simulation.

        Args:
            agents: List of trading agents
            num_rounds: Number of trading rounds
            parallel: Whether to run LLM agents in parallel
            replenish_endowments: If True, reset endowments each round (Gode & Sunder style).
                                  If False, endowments persist across rounds (exhaustion dynamics).
        """
        self.agents = agents
        self.market = DoubleAuction()
        self.num_rounds = num_rounds
        self.round_data = []
        self.round_results = []  # For DatasetExporter compatibility
        self.parallel = parallel
        self.replenish_endowments = replenish_endowments

        # Store initial endowments for replenishment
        self._initial_endowments = {
            agent.state.id: dict(agent.state.endowment) for agent in agents
        }

    def run(self):
        """Run simulation - uses async if agents support it and parallel=True."""
        # Check if we have LLM agents that support async
        has_llm_agents = any(isinstance(a, (ReactAgent, CoTAgent)) for a in self.agents)

        if self.parallel and has_llm_agents:
            return asyncio.run(self._run_async())
        else:
            return self._run_sync()

    def _run_sync(self):
        """Synchronous execution - one agent at a time."""
        for round_num in range(1, self.num_rounds + 1):
            print(f"Round {round_num}")
            market_info = self.market.get_market_info()
            market_info["round"] = round_num

            # Collect orders from all agents
            for agent in self.agents:
                order = agent.decide(market_info)
                if order:
                    self.market.submit(order)

            self._finish_round(round_num)

        return self.round_data

    async def _run_async(self):
        """Async execution - all agents decide in parallel."""
        for round_num in range(1, self.num_rounds + 1):
            print(f"Round {round_num}")
            market_info = self.market.get_market_info()
            market_info["round"] = round_num

            # Collect orders from all agents IN PARALLEL
            async def get_order(agent):
                if hasattr(agent, "decide_async"):
                    return await agent.decide_async(market_info)
                else:
                    return agent.decide(market_info)

            orders = await asyncio.gather(*[get_order(a) for a in self.agents])

            for order in orders:
                if order:
                    self.market.submit(order)

            self._finish_round(round_num)

        return self.round_data

    def _finish_round(self, round_num: int):
        """Clear market and update agents after a round."""
        # Capture bid-ask spread BEFORE clearing
        bid_ask_spread = self.market.get_bid_ask_spread()

        trades = self.market.clear()

        # Update agents with trade results and mark as traded
        for trade in trades:
            for agent in self.agents:
                if agent.state.id in [trade.buyer_id, trade.seller_id]:
                    agent.update({"trade": trade, "round": round_num})
                    # Mark ZI agents as having traded this round
                    if isinstance(agent, ZIAgent):
                        agent.mark_traded()

        # Log data
        round_data = {
            "round": round_num,
            "num_trades": len(trades),
            "prices": [t.price for t in trades],
            "volume": sum(t.quantity for t in trades),
            "bid_ask_spread": bid_ask_spread,
            "trades": trades,  # Keep trade objects for DatasetExporter
        }
        self.round_data.append(round_data)
        self.round_results.append(round_data)  # Alias for DatasetExporter

        # Prepare for next round
        self._prepare_next_round()

    def get_agent_logs(self) -> dict:
        """
        Extract decision logs grouped by agent.

        Returns:
            Dict mapping agent_id to list of decision logs.
        """
        logs = {}
        for agent in self.agents:
            logs[agent.state.id] = agent.get_logs()
        return logs

    def get_logs_by_round(self) -> list:
        """
        Extract decision logs grouped by round.

        Returns:
            List of dicts, each containing round number and all agent decisions.
            More intuitive for analyzing "what happened in round N".
        """
        # Collect all logs with agent metadata
        all_logs = []
        for agent in self.agents:
            for log in agent.get_logs():
                log_dict = log if isinstance(log, dict) else log
                log_dict["role"] = agent.state.role
                log_dict["valuation"] = agent.state.valuation.get("good_A", 0)
                all_logs.append(log_dict)

        # Group by round
        rounds = {}
        for log in all_logs:
            round_num = log.get("round", 0)
            if round_num not in rounds:
                rounds[round_num] = []
            rounds[round_num].append(log)

        # Convert to sorted list
        return [
            {"round": r, "decisions": rounds[r]}
            for r in sorted(rounds.keys())
        ]

    def _prepare_next_round(self):
        """Prepare agents for the next round."""
        for agent in self.agents:
            # Replenish endowments (Gode & Sunder style)
            if self.replenish_endowments:
                agent.state.endowment = dict(self._initial_endowments[agent.state.id])

            # Reset ZI agents' traded flag for new round
            if isinstance(agent, ZIAgent):
                agent.reset_for_new_round()

    def get_logs_flat(self) -> list:
        """
        Extract decision logs as a flat list for DataFrame analysis.

        Returns:
            List of dicts, one per agent-round, ready for pd.DataFrame().
            Includes agent metadata (role, valuation) in each row.
        """
        flat = []
        for agent in self.agents:
            for log in agent.get_logs():
                log_dict = log if isinstance(log, dict) else log
                row = {
                    "agent_id": log_dict.get("agent_id"),
                    "round": log_dict.get("round"),
                    "role": agent.state.role,
                    "valuation": agent.state.valuation.get("good_A", 0),
                    "agent_type": log_dict.get("agent_type"),
                    "action": log_dict.get("decision", {}).get("action"),
                    "price": log_dict.get("decision", {}).get("price"),
                    "quantity": log_dict.get("decision", {}).get("quantity"),
                    "order_submitted": log_dict.get("order") is not None,
                    "observation": log_dict.get("observation", ""),
                    "analysis": log_dict.get("analysis", ""),
                    "reasoning": log_dict.get("reasoning", ""),
                }
                flat.append(row)
        return flat


def create_agents_from_scheme(
    n_agents: int,
    agent_type: str = "react",
    provider: str = "deepseek",
    model: str = None,
    valuation_scheme: Optional[ValuationScheme] = None,
    force_participation: bool = True,
    zi_max_price: Optional[float] = None,
) -> List[BaseAgent]:
    """
    Create agents for the simulation using the new registry.

    Args:
        n_agents: Total number of agents (split 50/50 between buyers and sellers)
        agent_type: One of:
            - "zi": Zero-Intelligence (Gode & Sunder 1993 random baseline)
            - "react": LLM with simple reactive reasoning (observe → act)
            - "cot": LLM with Chain-of-Thought (observe → analyze → reason → act)
        provider: LLM provider ("deepseek", "anthropic", "openai", "google")
        model: Model name (defaults to provider's default)
        valuation_scheme: Valuation assignment strategy (defaults to LinearValuationScheme)
        force_participation: If True, LLM agents must submit orders (no hold option).
                            If False, agents can strategically choose to hold.
                            ZI agents always participate (single trade per round).
        zi_max_price: Upper bound for ZI seller asks. If None, auto-calculated as
                     max(buyer_valuations) + buffer to ensure equilibrium is tradeable.

    Returns:
        List of agents with heterogeneous valuations to enable gains from trade.
    """
    # Use default linear scheme if none provided
    if valuation_scheme is None:
        valuation_scheme = LinearValuationScheme()

    # Generate agent profiles from valuation scheme
    n_buyers = n_agents // 2
    n_sellers = n_agents - n_buyers
    profiles = valuation_scheme.generate_profiles(n_buyers, n_sellers)

    # Calculate max_price for ZI sellers if not provided
    # Using min(buyer_valuations) creates good bid/ask overlap for call markets
    if zi_max_price is None:
        buyer_valuations = [p.valuation for p in profiles if p.role == "buyer"]
        zi_max_price = min(buyer_valuations) if buyer_valuations else 60.0

    agents = []
    for i, profile in enumerate(profiles):
        state = AgentState(
            id=f"agent_{i}",
            endowment=profile.endowment,
            valuation={"good_A": profile.valuation},
            role=profile.role
        )

        # Use registry to create agents
        if agent_type == "zi":
            agents.append(create_agent("zi", state, max_price=zi_max_price))
        elif agent_type == "react":
            agents.append(create_agent("react", state, provider=provider, model=model,
                                       force_participation=force_participation))
        elif agent_type == "cot":
            agents.append(create_agent("cot", state, provider=provider, model=model,
                                       force_participation=force_participation))
        else:
            # Try registry for custom agent types
            agents.append(create_agent(agent_type, state, provider=provider, model=model))

    return agents
