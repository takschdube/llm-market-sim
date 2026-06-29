"""Agent specifications.

An `AgentSpec` is a declarative description of one or more agents that
share a kind, a strategy, and lineage metadata. `Population` aggregates
many `AgentSpec`s into a heterogeneous population. The runner expands a
spec into concrete engine agents by calling `_to_engine_agents`.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .strategies import (
    CustomStrategy,
    LLMPromptStrategy,
    StrategySpec,
    ZIStrategy,
)


class AgentSpec(BaseModel):
    """A specification of `count` identical agents.

    Two `AgentSpec`s with the same fields produce two distinct populations
    of agents, distinguishable in traces by `agent_id` (assigned by the
    runner) but identical in strategy and lineage.

    Attributes
    ----------
    kind
        "llm" for LLM-backed agents, "classical" for non-LLM strategies,
        "custom" for `CustomStrategy`-wrapped callables.
    strategy
        The decision rule. Carries strategy-specific parameters.
    count
        Number of independent agents instantiated from this spec.
    provider
        LLM provider name (e.g., "anthropic", "openai"). Required for
        kind="llm".
    model
        Specific model identifier. If omitted, the engine's provider
        default is used.
    lineage_tag
        Free-form lineage label such as
        "anthropic/claude-sonnet-4.6/base". Used by Governance to compute
        the architectural-diversity score and by Metric.icc_within with
        level="lineage".
    persona
        Optional persona string passed into LLM prompts.
    """

    kind: Literal["llm", "classical", "custom"]
    strategy: StrategySpec
    count: int = Field(default=1, ge=1)
    provider: Optional[str] = None
    model: Optional[str] = None
    lineage_tag: Optional[str] = None
    persona: Optional[str] = None

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def _to_engine_agents(self, profiles: Sequence[Any]) -> List[Any]:
        """Build engine agents from this spec and a sequence of profiles.

        Each profile carries (role, valuation, endowment). The method is
        called by the StudyRunner; it is not part of the public surface.
        """
        from src.agents.base import AgentState
        from src.agents.registry import create_agent

        engine_agents: List[Any] = []
        for idx, profile in enumerate(profiles):
            state = AgentState(
                id=f"{self.kind}_{self.strategy.name}_{idx}",
                endowment=dict(profile.endowment),
                valuation={"good_A": profile.valuation},
                role=profile.role,
            )

            if isinstance(self.strategy, ZIStrategy):
                kwargs: Dict[str, Any] = {}
                if self.strategy.max_price is not None:
                    kwargs["max_price"] = self.strategy.max_price
                engine_agents.append(create_agent("zi", state, **kwargs))

            elif isinstance(self.strategy, LLMPromptStrategy):
                if self.provider is None:
                    raise ValueError(
                        "LLM strategy requires AgentSpec.provider to be set"
                    )
                agent_type = "react" if self.strategy.reasoning == "react" else "cot"
                engine_agents.append(
                    create_agent(
                        agent_type,
                        state,
                        provider=self.provider,
                        model=self.model,
                        force_participation=self.strategy.force_participation,
                    )
                )

            elif isinstance(self.strategy, CustomStrategy):
                # Custom strategies wrap a callable; we adapt it to BaseAgent
                # via a small shim. Defined here to avoid a circular import.
                from src.agents.base import BaseAgent, DecisionLog, Order

                decide_fn = self.strategy.decide_fn
                strategy_name = self.strategy.name

                class _CallableAgent(BaseAgent):
                    def decide(self_inner, market_info: Dict) -> Optional[Order]:
                        result = decide_fn(self_inner.state, market_info)
                        log = DecisionLog(
                            round=market_info.get("round", 0),
                            agent_id=self_inner.state.id,
                            agent_type=f"custom_{strategy_name}",
                            decision={} if result is None else result.__dict__,
                            order=result,
                        )
                        self_inner.decision_logs.append(log)
                        return result

                engine_agents.append(_CallableAgent(state))

            else:
                raise NotImplementedError(
                    f"Strategy {self.strategy.name!r} not yet wired to engine; "
                    f"add a branch in AgentSpec._to_engine_agents."
                )

        return engine_agents


class Agent:
    """Namespace for agent builders.

    Three builders cover the three kinds: `Agent.llm` for LLM-backed
    agents (provider required), `Agent.classical` for non-LLM strategies
    (ZI, momentum, mean-reversion), and `Agent.custom` for callable-backed
    strategies.

    The `count` parameter is the number of identical agents this spec
    produces in a Population.
    """

    @staticmethod
    def llm(
        provider: str,
        strategy: StrategySpec,
        model: Optional[str] = None,
        count: int = 1,
        lineage_tag: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> AgentSpec:
        return AgentSpec(
            kind="llm",
            strategy=strategy,
            count=count,
            provider=provider,
            model=model,
            lineage_tag=lineage_tag,
            persona=persona,
        )

    @staticmethod
    def classical(strategy: StrategySpec, count: int = 1) -> AgentSpec:
        return AgentSpec(kind="classical", strategy=strategy, count=count)

    @staticmethod
    def custom(strategy: CustomStrategy, count: int = 1) -> AgentSpec:
        return AgentSpec(kind="custom", strategy=strategy, count=count)
