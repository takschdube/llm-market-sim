# src/agents/__init__.py
"""
Agent implementations for market simulation.

This module provides:
- Core abstractions (BaseAgent, AgentState, Order, DecisionLog)
- Agent registry for dynamic agent discovery
- Agent implementations (ZIAgent, ReactAgent, CoTAgent)
- LLM client for multi-provider support
- Prompt templates for LLM agents

Quick Start:
    from src.agents import create_agent, AgentState

    state = AgentState(id="agent_0", endowment={"good_A": 1, "money": 100},
                       valuation={"good_A": 15}, role="buyer")
    agent = create_agent("react", state, provider="deepseek")
    order = agent.decide(market_info)

Available Agents:
    - "zi": Zero-Intelligence baseline (Gode & Sunder 1993)
    - "react": Single-call LLM agent (observe → decide)
    - "cot": Chain-of-thought LLM agent (observe → analyze → reason → decide)

Adding New Agents:
    from src.agents import register_agent, BaseAgent

    @register_agent("my_agent")
    class MyAgent(BaseAgent):
        def decide(self, market_info):
            ...
"""

# Core abstractions
from .base import BaseAgent, AgentState, Order, DecisionLog

# Registry
from .registry import (
    AGENT_REGISTRY,
    register_agent,
    create_agent,
    list_agents,
    get_agent_class,
)

# Agent implementations
from .zi_agent import ZIAgent, ZeroIntelligenceAgent
from .react_agent import ReactAgent
from .cot_agent import CoTAgent

# LLM infrastructure
from .llm_client import LLMClient, LLMProvider, DEFAULT_MODELS, ALTERNATIVE_MODELS
from .response_parser import parse_json_response, parse_structured_response, decision_to_order

# Prompt system
from .prompts import (
    PromptContext,
    PromptTemplate,
    ReactPromptTemplate,
    CoTPromptTemplate,
    get_prompt_template,
    list_templates,
    export_all_templates,
    PROMPT_TEMPLATES,
)


__all__ = [
    # Core abstractions
    "BaseAgent",
    "AgentState",
    "Order",
    "DecisionLog",
    # Registry
    "AGENT_REGISTRY",
    "register_agent",
    "create_agent",
    "list_agents",
    "get_agent_class",
    # Agent implementations
    "ZIAgent",
    "ZeroIntelligenceAgent",  # Alias for backward compatibility
    "ReactAgent",
    "CoTAgent",
    # LLM infrastructure
    "LLMClient",
    "LLMProvider",
    "DEFAULT_MODELS",
    "ALTERNATIVE_MODELS",
    "parse_json_response",
    "parse_structured_response",
    "decision_to_order",
    # Prompt system
    "PromptContext",
    "PromptTemplate",
    "ReactPromptTemplate",
    "CoTPromptTemplate",
    "get_prompt_template",
    "list_templates",
    "export_all_templates",
    "PROMPT_TEMPLATES",
]
