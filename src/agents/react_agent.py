# src/agents/react_agent.py
"""
Reactive LLM Trading Agent
==========================
Single-call LLM agent: observe market → decide action.

This is the simplest LLM agent mode. It makes one LLM call per decision,
asking the model to observe the market state and output a trading decision.

Good for:
- Fast iteration (low latency)
- Cost-efficient experiments
- Baseline LLM performance measurement
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import BaseAgent, AgentState, Order, DecisionLog
from .registry import register_agent
from .llm_client import LLMClient, LLMProvider
from .prompts import PromptContext, ReactPromptTemplate
from .response_parser import parse_json_response, decision_to_order


@register_agent("react")
class ReactAgent(BaseAgent):
    """
    Single-call reactive LLM trading agent.

    Makes one LLM call per decision: observe → decide.
    Uses ReactPromptTemplate for prompt generation.

    Args:
        state: AgentState with role, valuation, and endowment
        provider: LLM provider ("deepseek", "anthropic", "openai", "google")
        model: Model name (optional, uses provider default)
        force_participation: If True, agent must submit order (no holding)

    Example:
        agent = ReactAgent(state, provider="anthropic", model="claude-sonnet-4-5-20250929")
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

        # Initialize LLM client and prompt template
        self.llm = LLMClient(provider, model)
        self.prompt_template = ReactPromptTemplate()

    def _build_context(self, market_info: Dict) -> PromptContext:
        """Build PromptContext from agent state and market info."""
        return PromptContext(
            agent_id=self.state.id,
            role=self.state.role,
            valuation=self.state.valuation.get("good_A", 0),
            endowment=self.state.endowment.copy(),
            round_num=market_info.get("round", 1),
            last_price=market_info.get("last_price"),
            bids=market_info.get("bids", []),
            asks=market_info.get("asks", []),
            trade_history=self.history.copy(),
        )

    def decide(self, market_info: Dict) -> Optional[Order]:
        """
        Make a trading decision via single LLM call.

        Args:
            market_info: Dict with round, bids, asks, last_price

        Returns:
            Order to submit, or None to hold
        """
        # Build context for prompt generation
        ctx = self._build_context(market_info)

        # Generate prompts
        system_prompt = self.prompt_template.system_prompt(ctx)
        user_prompt = self.prompt_template.user_prompt(ctx)

        # Make LLM call
        raw_response = self.llm.call(system_prompt, user_prompt, max_tokens=256)

        # Parse response
        decision = parse_json_response(raw_response)
        order = decision_to_order(decision, self.state.id)

        # Log the decision
        log = DecisionLog(
            round=market_info.get("round", 0),
            agent_id=self.state.id,
            agent_type="react",
            provider=self.provider,
            model=self.llm.model,
            market_state=market_info,
            observation=f"Round {ctx.round_num}, last_price={ctx.last_price}",
            decision=decision,
            raw_response=raw_response,
            order=order,
        )
        self.decision_logs.append(log)

        return order

    async def decide_async(self, market_info: Dict) -> Optional[Order]:
        """
        Make a trading decision via async LLM call (for parallel execution).

        Args:
            market_info: Dict with round, bids, asks, last_price

        Returns:
            Order to submit, or None to hold
        """
        # Build context for prompt generation
        ctx = self._build_context(market_info)

        # Generate prompts
        system_prompt = self.prompt_template.system_prompt(ctx)
        user_prompt = self.prompt_template.user_prompt(ctx)

        # Make async LLM call
        raw_response = await self.llm.call_async(system_prompt, user_prompt, max_tokens=256)

        # Parse response
        decision = parse_json_response(raw_response)
        order = decision_to_order(decision, self.state.id)

        # Log the decision
        log = DecisionLog(
            round=market_info.get("round", 0),
            agent_id=self.state.id,
            agent_type="react",
            provider=self.provider,
            model=self.llm.model,
            market_state=market_info,
            observation=f"Round {ctx.round_num}, last_price={ctx.last_price}",
            decision=decision,
            raw_response=raw_response,
            order=order,
        )
        self.decision_logs.append(log)

        return order

    def __repr__(self) -> str:
        return f"ReactAgent(id={self.state.id}, provider={self.provider})"
