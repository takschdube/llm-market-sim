# src/agents/registry.py
"""
Agent registry for dynamic agent discovery and instantiation.

This module provides a simple registry pattern for agents, similar to
PyTorch/HuggingFace model registries. New agent types can be registered
using the @register_agent decorator and instantiated via create_agent().

Example:
    @register_agent("my_agent")
    class MyAgent(BaseAgent):
        def decide(self, market_info):
            ...

    # Later:
    agent = create_agent("my_agent", state, **kwargs)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from .base import AgentState, BaseAgent

# Global registry mapping agent names to classes
AGENT_REGISTRY: Dict[str, Type["BaseAgent"]] = {}


def register_agent(name: str):
    """
    Decorator to register an agent class.

    Args:
        name: String identifier for the agent type (e.g., "zi", "react", "cot")

    Example:
        @register_agent("dqn")
        class DQNAgent(BaseAgent):
            ...
    """
    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        if name in AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered")
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def create_agent(name: str, state: "AgentState", **kwargs) -> "BaseAgent":
    """
    Factory function to create agents by name.

    Args:
        name: Registered agent type (e.g., "zi", "react", "cot")
        state: AgentState for the new agent
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        Instantiated agent

    Raises:
        ValueError: If agent type is not registered
    """
    if name not in AGENT_REGISTRY:
        available = list(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: '{name}'. Available: {available}")
    return AGENT_REGISTRY[name](state, **kwargs)


def list_agents() -> List[str]:
    """
    List all registered agent types.

    Returns:
        List of registered agent names
    """
    return list(AGENT_REGISTRY.keys())


def get_agent_class(name: str) -> Type["BaseAgent"]:
    """
    Get agent class by name without instantiating.

    Args:
        name: Registered agent type

    Returns:
        Agent class

    Raises:
        ValueError: If agent type is not registered
    """
    if name not in AGENT_REGISTRY:
        available = list(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: '{name}'. Available: {available}")
    return AGENT_REGISTRY[name]
