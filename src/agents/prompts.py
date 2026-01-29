# src/agents/prompts.py
"""
Modular Prompt Management System
================================
Centralized prompt templates for all LLM-based trading agents.

This module provides:
1. Structured prompt templates for different reasoning modes
2. Configurable prompt parameters
3. Full auditability - all prompts can be logged/exported
4. Easy experimentation with prompt variations

Prompt Design Philosophy:
- Clear role assignment (buyer/seller)
- Explicit constraints (budget, valuation)
- Structured output format (JSON)
- Minimal ambiguity to reduce parsing errors

References:
- Wei et al. (2022) Chain-of-Thought Prompting
- Yao et al. (2023) ReAct: Synergizing Reasoning and Acting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class PromptContext:
    """
    All context needed to generate a trading prompt.

    This captures the agent's state and market information in a
    structured format that can be passed to any prompt template.
    """
    # Agent identity
    agent_id: str
    role: str  # "buyer" or "seller"

    # Agent state
    valuation: float
    endowment: Dict[str, float]

    # Market state
    round_num: int
    last_price: Optional[float] = None
    bids: List[float] = field(default_factory=list)
    asks: List[float] = field(default_factory=list)
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None

    # History
    trade_history: List[Dict] = field(default_factory=list)

    # Optional persona/style
    persona: str = "rational"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "valuation": self.valuation,
            "endowment": self.endowment,
            "round_num": self.round_num,
            "last_price": self.last_price,
            "bids": self.bids,
            "asks": self.asks,
            "persona": self.persona,
        }


class PromptTemplate(ABC):
    """
    Abstract base class for prompt templates.

    Each template defines how to generate system and user prompts
    for a specific reasoning mode.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Template identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @abstractmethod
    def system_prompt(self, ctx: PromptContext) -> str:
        """Generate the system/instruction prompt."""
        pass

    @abstractmethod
    def user_prompt(self, ctx: PromptContext) -> str:
        """Generate the user/input prompt."""
        pass

    def to_dict(self) -> Dict[str, str]:
        """Export template metadata for documentation."""
        return {
            "name": self.name,
            "description": self.description,
        }


# =============================================================================
# REACT MODE PROMPTS
# =============================================================================

class ReactPromptTemplate(PromptTemplate):
    """
    Simple reactive prompt - observe and act in one step.

    Design principles:
    - Minimal context to reduce latency
    - Clear role identification
    - Explicit JSON format requirement
    - Direct instruction to submit an order
    """

    @property
    def name(self) -> str:
        return "react_v1"

    @property
    def description(self) -> str:
        return "Simple reactive trading prompt. Single-step observe-and-act."

    def system_prompt(self, ctx: PromptContext) -> str:
        if ctx.role == "buyer":
            role_instruction = (
                f"You are a BUYER. Your goal is to BUY good_A at a price "
                f"BELOW your valuation of {ctx.valuation}. "
                f"Submit a BUY order with a bid price."
            )
        else:
            role_instruction = (
                f"You are a SELLER. Your goal is to SELL good_A at a price "
                f"ABOVE your valuation of {ctx.valuation}. "
                f"Submit a SELL order with an ask price."
            )

        return f"""You are a trader in a double auction market simulation.

{role_instruction}

Your current holdings: {ctx.endowment}
Your valuation for good_A: {ctx.valuation}

RULES:
- Buyers profit when buy_price < valuation
- Sellers profit when sell_price > valuation
- You MUST submit an order (no holding allowed)

Respond with ONLY valid JSON in this format:
{{"action": "buy" or "sell", "price": <number>, "quantity": 1}}"""

    def user_prompt(self, ctx: PromptContext) -> str:
        last_price_str = str(ctx.last_price) if ctx.last_price else "None (no trades yet)"

        return f"""Round {ctx.round_num}

Market State:
- Last transaction price: {last_price_str}
- Current bids (buy orders): {ctx.bids if ctx.bids else "None"}
- Current asks (sell orders): {ctx.asks if ctx.asks else "None"}

Your decision (JSON only):"""


# =============================================================================
# CHAIN-OF-THOUGHT (COT) MODE PROMPTS
# =============================================================================

class CoTPromptTemplate(PromptTemplate):
    """
    Chain-of-thought prompt with explicit reasoning steps.

    Design principles:
    - Structured reasoning (observe → analyze → reason → decide)
    - Each step builds on previous
    - Full auditability of reasoning process
    - Balanced between thoroughness and token efficiency
    """

    @property
    def name(self) -> str:
        return "cot_v1"

    @property
    def description(self) -> str:
        return "Chain-of-thought trading prompt with explicit reasoning steps."

    def system_prompt(self, ctx: PromptContext) -> str:
        return f"""You are a rational trader in an economic market simulation.

Your Profile:
- Role: {ctx.role.upper()}
- Valuation for good_A: {ctx.valuation}
- Current holdings: {ctx.endowment}
- Trading persona: {ctx.persona}

Economic Principles:
- As a {'BUYER' if ctx.role == 'buyer' else 'SELLER'}, you profit when you {'buy BELOW' if ctx.role == 'buyer' else 'sell ABOVE'} your valuation
- Your surplus = {'valuation - price' if ctx.role == 'buyer' else 'price - valuation'}
- Only trade when expected surplus is positive

You will analyze the market step by step, then make a decision.

Output format (JSON):
{{
    "analysis": "Brief market analysis (1-2 sentences)",
    "reasoning": "Your strategic reasoning (2-3 sentences)",
    "decision": {{"action": "buy"|"sell"|"hold", "price": <number>, "quantity": 1}}
}}"""

    def user_prompt(self, ctx: PromptContext) -> str:
        last_price_str = str(ctx.last_price) if ctx.last_price else "No trades yet"
        spread_str = f"{ctx.spread:.2f}" if ctx.spread else "N/A"

        # Format recent history
        history_str = self._format_history(ctx.trade_history, ctx.agent_id)

        return f"""=== ROUND {ctx.round_num} ===

MARKET STATE:
- Last transaction price: {last_price_str}
- Best bid (highest buy): {ctx.best_bid if ctx.best_bid else 'None'}
- Best ask (lowest sell): {ctx.best_ask if ctx.best_ask else 'None'}
- Bid-ask spread: {spread_str}
- All bids: {ctx.bids if ctx.bids else 'None'}
- All asks: {ctx.asks if ctx.asks else 'None'}

YOUR SITUATION:
- Role: {ctx.role.upper()}
- Valuation: {ctx.valuation}
- Holdings: {ctx.endowment}
{history_str}

INSTRUCTIONS:
1. Analyze the current market conditions
2. Reason about profitable trading opportunities
3. Make your trading decision

Provide your analysis, reasoning, and decision in JSON format."""

    def _format_history(self, history: List[Dict], agent_id: str) -> str:
        """Format trade history for the prompt."""
        if not history:
            return "- Trade history: No trades yet"

        recent = history[-3:]  # Last 3 trades
        lines = ["- Recent trades:"]
        for h in recent:
            round_num = h.get("round", "?")
            price = h.get("price", "?")
            role = h.get("role", "traded")
            lines.append(f"  Round {round_num}: {role} at price {price}")

        return "\n".join(lines)


# =============================================================================
# COT STEP-BY-STEP PROMPTS (for multi-call CoT)
# =============================================================================

class CoTObservePrompt(PromptTemplate):
    """Observation step prompt for multi-call CoT."""

    @property
    def name(self) -> str:
        return "cot_observe_v1"

    @property
    def description(self) -> str:
        return "Observation step for chain-of-thought reasoning."

    def system_prompt(self, ctx: PromptContext) -> str:
        return "You are observing a market. Summarize the current state clearly and concisely."

    def user_prompt(self, ctx: PromptContext) -> str:
        return f"""Round {ctx.round_num} of trading.

Market state:
- Last price: {ctx.last_price if ctx.last_price else 'No trades yet'}
- Current bids: {ctx.bids if ctx.bids else 'None'}
- Current asks: {ctx.asks if ctx.asks else 'None'}

My situation:
- Role: {ctx.role}
- Holdings: {ctx.endowment}
- My valuation: {ctx.valuation}

Summarize what you observe about the market."""


class CoTAnalyzePrompt(PromptTemplate):
    """Analysis step prompt for multi-call CoT."""

    @property
    def name(self) -> str:
        return "cot_analyze_v1"

    @property
    def description(self) -> str:
        return "Analysis step for chain-of-thought reasoning."

    def system_prompt(self, ctx: PromptContext) -> str:
        return "You are analyzing a market. Be concise and factual."

    def user_prompt(self, ctx: PromptContext, observation: str = "") -> str:
        return f"""{observation}

Briefly analyze the current market conditions (2-3 sentences).
What opportunities or risks do you see?"""


class CoTReasonPrompt(PromptTemplate):
    """Reasoning step prompt for multi-call CoT."""

    @property
    def name(self) -> str:
        return "cot_reason_v1"

    @property
    def description(self) -> str:
        return "Strategic reasoning step for chain-of-thought."

    def system_prompt(self, ctx: PromptContext) -> str:
        return "You are reasoning about trading strategy. Be logical and concise."

    def user_prompt(self, ctx: PromptContext, observation: str = "", analysis: str = "") -> str:
        return f"""Market observation:
{observation}

Your analysis:
{analysis}

Given your valuation of {ctx.valuation} for good_A,
what is your strategic reasoning? Consider:
- What price would be profitable for you?
- What action makes sense given current market conditions?

Explain your reasoning in 2-3 sentences."""


class CoTDecidePrompt(PromptTemplate):
    """Decision step prompt for multi-call CoT."""

    @property
    def name(self) -> str:
        return "cot_decide_v1"

    @property
    def description(self) -> str:
        return "Decision step for chain-of-thought reasoning."

    def system_prompt(self, ctx: PromptContext) -> str:
        return f"You are a trader. Holdings: {ctx.endowment}. Valuation: {ctx.valuation}. Respond with JSON only."

    def user_prompt(self, ctx: PromptContext, analysis: str = "", reasoning: str = "") -> str:
        return f"""Based on your analysis and reasoning:

Analysis: {analysis}
Reasoning: {reasoning}

Now make your trading decision. Respond with ONLY valid JSON:
{{"action": "buy"|"sell"|"hold", "price": <number>, "quantity": 1}}"""


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

# All available prompt templates
PROMPT_TEMPLATES = {
    "react": ReactPromptTemplate,
    "cot": CoTPromptTemplate,
    "cot_observe": CoTObservePrompt,
    "cot_analyze": CoTAnalyzePrompt,
    "cot_reason": CoTReasonPrompt,
    "cot_decide": CoTDecidePrompt,
}


def get_prompt_template(name: str) -> PromptTemplate:
    """Factory function to get a prompt template by name."""
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {name}. Available: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[name]()


def list_templates() -> List[Dict[str, str]]:
    """List all available prompt templates with descriptions."""
    return [
        {"name": name, "description": cls().description}
        for name, cls in PROMPT_TEMPLATES.items()
    ]


def export_all_templates(ctx: PromptContext) -> Dict[str, Dict[str, str]]:
    """
    Export all prompt templates with example prompts for documentation.

    This is useful for paper appendices and reproducibility.
    """
    result = {}
    for name, cls in PROMPT_TEMPLATES.items():
        template = cls()
        result[name] = {
            "description": template.description,
            "system_prompt": template.system_prompt(ctx),
            "user_prompt": template.user_prompt(ctx) if not name.startswith("cot_") or name == "cot" else "[Requires previous step output]",
        }
    return result
