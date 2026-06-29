"""Strategy specifications.

Strategies are first-class spec objects that an Agent carries. The
`Strategy` namespace exposes builders for the built-in strategies plus
`Strategy.from_callable` for one-off user strategies and
`Strategy.register` as a decorator to add new strategy types to the
registry.

Built-in strategies cover the two ends of the population: LLM-based
agents (`LLMPromptStrategy`) and classical baselines/heuristics
(`ZIStrategy`, `MomentumStrategy`, `MeanReversionStrategy`).
"""
from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field


class _BaseStrategy(BaseModel):
    """Internal base for strategy specs. Frozen, validated, no extras."""

    name: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ZIStrategy(_BaseStrategy):
    """Zero-Intelligence strategy (Gode & Sunder, 1993).

    Buyers bid uniformly in [0, valuation]; sellers ask uniformly in
    [cost, max_price]. The classical uncorrelated baseline.
    """

    name: Literal["zi"] = "zi"
    price_bounds: Literal["role_dependent", "fixed"] = "role_dependent"
    max_price: Optional[float] = None


class LLMPromptStrategy(_BaseStrategy):
    """LLM-based bidding strategy with prompt template and reasoning mode.

    The runner pairs this spec with the Agent's `provider` and `model`
    fields to produce an engine LLM agent (Engine `ReactAgent` or
    `CoTAgent`).
    """

    name: Literal["llm_prompt"] = "llm_prompt"
    reasoning: Literal["react", "cot"] = "react"
    prompt_version: str = "v1"
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    persona: Optional[str] = None
    force_participation: bool = True


class MomentumStrategy(_BaseStrategy):
    """Momentum strategy based on recent clearing-price changes."""

    name: Literal["momentum"] = "momentum"
    window: int = Field(default=5, ge=1)
    beta: float = Field(default=0.3, ge=0.0, le=1.0)


class MeanReversionStrategy(_BaseStrategy):
    """Mean-reversion strategy targeting a moving estimate of equilibrium."""

    name: Literal["mean_reversion"] = "mean_reversion"
    halflife: int = Field(default=10, ge=1)


class CustomStrategy(_BaseStrategy):
    """User-defined strategy backed by an arbitrary callable.

    The callable must accept `(state, market_info)` and return either an
    `Order`-shaped dict or `None`. Use this for one-off experiments. For
    strategies you want reusable, register a Pydantic subclass of
    `_BaseStrategy` via `Strategy.register(name)`.
    """

    name: str
    decide_fn: Callable[..., Any]

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


# Union covers all built-in spec types. User-registered strategies must
# subclass `_BaseStrategy` and will be accepted at runtime.
StrategySpec = Union[
    ZIStrategy,
    LLMPromptStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    CustomStrategy,
]


class _StrategyRegistry:
    """Internal registry of strategy spec types keyed by name.

    The registry is used by `Strategy.from_dict(...)` (for config-driven
    runs) and by `Trace.evaluate(...)` when deserializing traces.
    """

    _registry: ClassVar[Dict[str, Type[_BaseStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[_BaseStrategy]) -> None:
        if name in cls._registry:
            raise ValueError(f"Strategy {name!r} already registered")
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Type[_BaseStrategy]:
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(f"Strategy {name!r} not registered. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())


# Register the built-ins at import time.
_StrategyRegistry.register("zi", ZIStrategy)
_StrategyRegistry.register("llm_prompt", LLMPromptStrategy)
_StrategyRegistry.register("momentum", MomentumStrategy)
_StrategyRegistry.register("mean_reversion", MeanReversionStrategy)
_StrategyRegistry.register("custom", CustomStrategy)


class Strategy:
    """Namespace for strategy builders.

    Use the staticmethods to construct `StrategySpec` value objects:

    >>> zi = Strategy.zi()
    >>> claude = Strategy.llm_prompt(reasoning="cot", prompt_version="v2")
    >>> my_strategy = Strategy.from_callable("my_bid", fn=lambda s, m: ...)

    For permanent custom strategy types, subclass `_BaseStrategy` and
    register::

        @Strategy.register("aggressive_bidder")
        class AggressiveBidder(_BaseStrategy):
            name: Literal["aggressive_bidder"] = "aggressive_bidder"
            discount: float = 0.01
    """

    # Expose the types for static analysis / isinstance checks.
    ZI = ZIStrategy
    LLMPrompt = LLMPromptStrategy
    Momentum = MomentumStrategy
    MeanReversion = MeanReversionStrategy
    Custom = CustomStrategy

    @staticmethod
    def zi(
        price_bounds: str = "role_dependent",
        max_price: Optional[float] = None,
    ) -> ZIStrategy:
        return ZIStrategy(price_bounds=price_bounds, max_price=max_price)

    @staticmethod
    def llm_prompt(
        reasoning: str = "react",
        prompt_version: str = "v1",
        temperature: float = 1.0,
        persona: Optional[str] = None,
        force_participation: bool = True,
    ) -> LLMPromptStrategy:
        return LLMPromptStrategy(
            reasoning=reasoning,
            prompt_version=prompt_version,
            temperature=temperature,
            persona=persona,
            force_participation=force_participation,
        )

    @staticmethod
    def momentum(window: int = 5, beta: float = 0.3) -> MomentumStrategy:
        return MomentumStrategy(window=window, beta=beta)

    @staticmethod
    def mean_reversion(halflife: int = 10) -> MeanReversionStrategy:
        return MeanReversionStrategy(halflife=halflife)

    @staticmethod
    def from_callable(name: str, fn: Callable[..., Any]) -> CustomStrategy:
        """Wrap a one-off callable as a CustomStrategy."""
        return CustomStrategy(name=name, decide_fn=fn)

    @staticmethod
    def register(name: str):
        """Decorator to register a permanent custom strategy class."""

        def decorator(strategy_cls: Type[_BaseStrategy]) -> Type[_BaseStrategy]:
            _StrategyRegistry.register(name, strategy_cls)
            return strategy_cls

        return decorator

    @staticmethod
    def list_registered() -> list[str]:
        return _StrategyRegistry.list()
