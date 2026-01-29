# src/utils/prompts.py
"""
Prompt Templates
================
Centralized prompt management for LLM agents. Separating prompts
from agent logic enables:
- Easy A/B testing of different prompt strategies
- Reproducibility across experiments
- Clear documentation of agent instructions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""
    name: str
    system_template: str
    user_template: str
    description: str = ""

    def format_system(self, **kwargs: Any) -> str:
        """Format the system prompt with given variables."""
        return self.system_template.format(**kwargs)

    def format_user(self, **kwargs: Any) -> str:
        """Format the user prompt with given variables."""
        return self.user_template.format(**kwargs)


# =============================================================================
# MINIMAL AGENT PROMPTS
# =============================================================================

MINIMAL_SYSTEM = """You are a trader in a market simulation. Respond ONLY with valid JSON, no other text.

Your holdings: {endowment}
Your valuation for good_A: {valuation}

Rules:
- If you have money and want to buy, bid at or below your valuation
- If you have goods and want to sell, ask at or above your valuation
- You can also hold (do nothing)

Respond with exactly this JSON format:
{{"action": "buy", "good": "good_A", "price": 15.0, "quantity": 1}}
or {{"action": "sell", "good": "good_A", "price": 10.0, "quantity": 1}}
or {{"action": "hold", "good": "good_A", "price": 0, "quantity": 0}}"""

MINIMAL_USER = """Round {round}
Last price: {last_price}
Current bids: {bids}
Current asks: {asks}

Your decision (JSON only):"""

MINIMAL_PROMPT = PromptTemplate(
    name="minimal",
    system_template=MINIMAL_SYSTEM,
    user_template=MINIMAL_USER,
    description="Minimal prompt for direct LLM response without chain-of-thought",
)


# =============================================================================
# CHAIN-OF-THOUGHT (COT) AGENT PROMPTS
# =============================================================================

COT_SYSTEM = """You are a trader in an economic market simulation. Your goal is to maximize utility through strategic trading.

Your persona: {persona}

You must respond with a structured JSON containing your analysis, reasoning, and decision.

Current holdings: {endowment}
Your valuation for good_A: {valuation} (this is how much the good is worth TO YOU)

Trading rules:
- As a BUYER: You profit when you buy below your valuation. Never bid above your valuation.
- As a SELLER: You profit when you sell above your valuation. Never ask below your valuation.
- You can HOLD if no profitable trade is available.

Respond with this exact JSON structure:
{{
    "analysis": "Brief analysis of current market conditions (1-2 sentences)",
    "reasoning": "Your strategic reasoning for this decision (2-3 sentences)",
    "decision": {{
        "action": "buy" | "sell" | "hold",
        "good": "good_A",
        "price": <number>,
        "quantity": 1
    }}
}}"""

COT_USER = """Round {round} of trading.

MARKET STATE:
- Last transaction price: {last_price}
- Current buy orders (bids): {bids}
- Current sell orders (asks): {asks}

YOUR SITUATION:
- Holdings: {endowment}
- Your valuation for good_A: {valuation}
{history_summary}

Analyze the market, explain your reasoning, and make your trading decision."""

COT_PROMPT = PromptTemplate(
    name="cot",
    system_template=COT_SYSTEM,
    user_template=COT_USER,
    description="Chain-of-thought prompt requiring analysis, reasoning, then decision",
)


# =============================================================================
# EXPERIMENTAL PROMPTS FOR FUTURE RESEARCH
# =============================================================================

STRATEGIC_SYSTEM = """You are a sophisticated trader in an economic market simulation.

Your holdings: {endowment}
Your valuation for good_A: {valuation}

Advanced considerations:
1. Other traders may have different valuations - exploit this
2. Early rounds are for price discovery; later rounds for optimal trades
3. Consider the bid-ask spread and market depth
4. You can influence market prices with your orders

Respond with JSON including your strategy:
{{
    "market_assessment": "Is the market overvalued or undervalued?",
    "strategy": "Your strategic approach for this round",
    "decision": {{"action": "...", "good": "good_A", "price": ..., "quantity": 1}}
}}"""

STRATEGIC_PROMPT = PromptTemplate(
    name="strategic",
    system_template=STRATEGIC_SYSTEM,
    user_template=COT_USER,  # Reuse COT user prompt
    description="Advanced strategic reasoning prompt (experimental)",
)


COOPERATIVE_SYSTEM = """You are a trader in a market simulation. Your goal is market efficiency.

Your holdings: {endowment}
Your valuation for good_A: {valuation}

You aim to:
1. Trade at fair prices that benefit both parties
2. Contribute to price discovery
3. Help the market reach equilibrium

Respond with JSON:
{{
    "fair_price_estimate": "Your estimate of the fair market price",
    "decision": {{"action": "...", "good": "good_A", "price": ..., "quantity": 1}}
}}"""

COOPERATIVE_PROMPT = PromptTemplate(
    name="cooperative",
    system_template=COOPERATIVE_SYSTEM,
    user_template=MINIMAL_USER,
    description="Cooperative agent focused on market efficiency (experimental)",
)


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

PROMPTS: Dict[str, PromptTemplate] = {
    "minimal": MINIMAL_PROMPT,
    "cot": COT_PROMPT,
    "strategic": STRATEGIC_PROMPT,
    "cooperative": COOPERATIVE_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: One of "minimal", "cot", "strategic", "cooperative"

    Returns:
        The corresponding PromptTemplate.

    Raises:
        KeyError: If prompt name not found.
    """
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt: {name}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[name]


def list_prompts() -> Dict[str, str]:
    """List all available prompts with descriptions."""
    return {name: p.description for name, p in PROMPTS.items()}
