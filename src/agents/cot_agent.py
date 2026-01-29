# src/agents/cot_agent.py
"""
Chain-of-Thought LLM Trading Agent
==================================
Multi-step reasoning agent: observe → analyze → reason → decide.

Uses LangGraph to structure the reasoning process into explicit steps,
providing full auditability of the agent's decision-making process.

Good for:
- Research requiring reasoning traces
- Understanding agent behavior
- Comparing explicit vs implicit reasoning
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END

from .base import BaseAgent, AgentState, Order, DecisionLog
from .registry import register_agent
from .llm_client import LLMClient, LLMProvider
from .response_parser import parse_json_response, decision_to_order


@register_agent("cot")
class CoTAgent(BaseAgent):
    """
    Chain-of-Thought LLM trading agent with explicit reasoning steps.

    Uses LangGraph to structure the decision process:
    1. Observe: Summarize market state
    2. Analyze: Analyze market conditions
    3. Reason: Strategic reasoning about price
    4. Decide: Make final trading decision

    Each step is logged for research auditability.

    Args:
        state: AgentState with role, valuation, and endowment
        provider: LLM provider ("deepseek", "anthropic", "openai", "google")
        model: Model name (optional, uses provider default)
        force_participation: If True, agent must submit order (no holding)

    Example:
        agent = CoTAgent(state, provider="anthropic")
        order = agent.decide(market_info)
    """

    def __init__(
        self,
        state: AgentState,
        provider: LLMProvider = "deepseek",
        model: Optional[str] = None,
        force_participation: bool = True,
    ):
        super().__init__(state)
        self.provider = provider
        self.model = model
        self.force_participation = force_participation

        # Initialize LLM client
        self.llm = LLMClient(provider, model)

        # Build the reasoning graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for chain-of-thought reasoning."""
        builder = StateGraph(dict)

        # Add nodes for each reasoning step
        builder.add_node("observe", self._observe_node)
        builder.add_node("analyze", self._analyze_node)
        builder.add_node("reason", self._reason_node)
        builder.add_node("decide", self._decide_node)

        # Linear flow: observe → analyze → reason → decide
        builder.add_edge(START, "observe")
        builder.add_edge("observe", "analyze")
        builder.add_edge("analyze", "reason")
        builder.add_edge("reason", "decide")
        builder.add_edge("decide", END)

        return builder.compile()

    def _observe_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Observe and summarize market state."""
        market = state["market_info"]
        role = self.state.role
        valuation = self.state.valuation.get("good_A", 0)

        # Build observation with role context
        if role == "buyer":
            role_context = f"I am a BUYER. I profit by buying below my valuation of {valuation}."
        else:
            role_context = f"I am a SELLER. I profit by selling above my valuation of {valuation}."

        observation = f"""Round {market.get('round', 1)} of trading.

My role: {role.upper()}
{role_context}

Market state:
- Last price: {market.get('last_price', 'No trades yet')}
- Current bids: {market.get('bids', [])}
- Current asks: {market.get('asks', [])}

My situation:
- Holdings: {state['endowment']}
- My valuation: {valuation}"""

        state["observation"] = observation
        state["role"] = role
        return state

    def _analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Analyze market conditions via LLM call."""
        role = state.get("role", self.state.role)
        valuation = self.state.valuation.get("good_A", 0)

        if role == "buyer":
            role_hint = f"You are a BUYER looking to buy below your valuation of {valuation}."
        else:
            role_hint = f"You are a SELLER looking to sell above your valuation of {valuation}."

        prompt = f"""{state['observation']}

{role_hint}

Briefly analyze the current market conditions (2-3 sentences).
What opportunities or risks do you see for your role?"""

        response = self.llm.call(
            system=f"You are a {role.upper()} analyzing a market. Focus on opportunities relevant to your role. Be concise.",
            user=prompt,
            max_tokens=200
        )

        state["analysis"] = response.strip()
        return state

    def _reason_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Strategic reasoning via LLM call."""
        role = state.get("role", self.state.role)
        valuation = self.state.valuation.get("good_A", 0)

        if role == "buyer":
            action_constraint = f"As a BUYER, you can ONLY submit BUY orders. You profit by buying below {valuation}."
        else:
            action_constraint = f"As a SELLER, you can ONLY submit SELL orders. You profit by selling above {valuation}."

        prompt = f"""Market observation:
{state['observation']}

Your analysis:
{state['analysis']}

{action_constraint}

Given your valuation of {valuation} for good_A, what is your strategic reasoning?
- What price would be profitable for your {role.upper()} role?
- You can ONLY {role.upper()} - what price makes sense?

Explain your reasoning in 2-3 sentences."""

        response = self.llm.call(
            system=f"You are a {role.upper()} reasoning about trading strategy. Be logical and concise.",
            user=prompt,
            max_tokens=200
        )

        state["reasoning"] = response.strip()
        return state

    def _decide_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Make final decision via LLM call."""
        role = self.state.role
        valuation = self.state.valuation.get("good_A", 0)

        if role == "buyer":
            action_options = '"buy"' if self.force_participation else '"buy"|"hold"'
            role_instruction = f"You are a BUYER. Submit a BUY order below your valuation of {valuation}."
        else:
            action_options = '"sell"' if self.force_participation else '"sell"|"hold"'
            role_instruction = f"You are a SELLER. Submit a SELL order above your valuation of {valuation}."

        if self.force_participation:
            participation_note = "You MUST submit an order."
        else:
            participation_note = "You can HOLD if no profitable opportunity exists."

        prompt = f"""Based on your analysis and reasoning:

Analysis: {state['analysis']}
Reasoning: {state['reasoning']}

{role_instruction}
{participation_note}

Make your trading decision. Respond with ONLY valid JSON:
{{"action": {action_options}, "price": <number>, "quantity": 1}}"""

        raw_response = self.llm.call(
            system=f"You are a trader. Holdings: {self.state.endowment}. Valuation: {valuation}. Respond with JSON only.",
            user=prompt,
            max_tokens=100
        )

        state["raw_response"] = raw_response

        # Parse decision
        decision = parse_json_response(raw_response)
        state["decision"] = decision
        state["order"] = decision_to_order(decision, self.state.id)

        return state

    def decide(self, market_info: Dict) -> Optional[Order]:
        """
        Make a trading decision via chain-of-thought reasoning.

        Args:
            market_info: Dict with round, bids, asks, last_price

        Returns:
            Order to submit, or None to hold
        """
        # Initialize state for graph
        initial_state = {
            "market_info": market_info,
            "agent_id": self.state.id,
            "endowment": self.state.endowment.copy(),
            "valuation": self.state.valuation.copy(),
            "history": self.history.copy(),
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Log the decision with full reasoning trace
        log = DecisionLog(
            round=market_info.get("round", 0),
            agent_id=self.state.id,
            agent_type="cot",
            provider=self.provider,
            model=self.llm.model,
            market_state=market_info,
            observation=result.get("observation", ""),
            analysis=result.get("analysis", ""),
            reasoning=result.get("reasoning", ""),
            decision=result.get("decision", {}),
            raw_response=result.get("raw_response", ""),
            order=result.get("order"),
        )
        self.decision_logs.append(log)

        return result.get("order")

    def __repr__(self) -> str:
        return f"CoTAgent(id={self.state.id}, provider={self.provider})"
