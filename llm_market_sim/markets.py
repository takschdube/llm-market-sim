"""Market mechanism specifications.

`MarketSpec` is a frozen Pydantic value object. The `Market` namespace
exposes builder factories that return `MarketSpec`s. Constructing a
`MarketSpec` does not run a market; calling `Study.run(...)` does.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValuationsSpec(BaseModel):
    """Specification of agent valuations.

    `kind` selects one of the four schemes in `src/simulation/valuations.py`.
    `params` is a free-form dict of scheme-specific parameters that the
    engine adapter forwards to the underlying ValuationScheme.
    """

    kind: Literal["linear", "uniform", "symmetric", "fixed"]
    n_buyers: int = Field(ge=1)
    n_sellers: int = Field(ge=1)
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_engine(self):
        """Build the corresponding engine ValuationScheme."""
        from src.simulation.valuations import (
            FixedValuationScheme,
            LinearValuationScheme,
            SymmetricValuationScheme,
            UniformValuationScheme,
        )

        cls = {
            "linear": LinearValuationScheme,
            "uniform": UniformValuationScheme,
            "symmetric": SymmetricValuationScheme,
            "fixed": FixedValuationScheme,
        }[self.kind]
        return cls(**self.params)


class ClearingSpec(BaseModel):
    """Specification of the clearing rule used to compute trade prices."""

    rule: Literal["uniform_price", "pairwise_midpoint"]
    tie_break: Literal["random", "first", "lowest_id"] = "random"

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Public names -> engine names. The public surface uses the short
    # forms (`uniform_price`, `pairwise_midpoint`); the engine spells the
    # first as `uniform_price_call` for historical reasons.
    _ENGINE_NAMES = {
        "uniform_price": "uniform_price_call",
        "pairwise_midpoint": "pairwise_midpoint",
    }

    @property
    def engine_mechanism_name(self) -> str:
        return self._ENGINE_NAMES[self.rule]


class MarketSpec(BaseModel):
    """Specification of a market mechanism.

    A `MarketSpec` is fully declarative. Two specs that compare equal will
    run identically given the same population and seed. Pass through
    `Study(market=spec, ...)` to execute.
    """

    type: Literal["call_auction", "cda"] = "call_auction"
    n_agents: int = Field(ge=2)
    rounds: int = Field(ge=1, default=20)
    valuations: ValuationsSpec
    clearing: ClearingSpec
    fee_bps: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("n_agents")
    @classmethod
    def _even_n_agents(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(
                f"n_agents must be even (split 50/50 buyer/seller); got {v}"
            )
        return v


class _Valuations:
    """Namespace for valuation builders.

    Exposed as `Market.valuations`. The methods return `ValuationsSpec`
    objects that compose into a `MarketSpec`.
    """

    @staticmethod
    def linear(
        n_each: int = 6,
        buyer_base: float = 60.0,
        buyer_step: float = 5.0,
        seller_base: float = 5.0,
        seller_step: float = 5.0,
    ) -> ValuationsSpec:
        """Smith-style linear induced values."""
        return ValuationsSpec(
            kind="linear",
            n_buyers=n_each,
            n_sellers=n_each,
            params={
                "buyer_base": buyer_base,
                "buyer_step": buyer_step,
                "seller_base": seller_base,
                "seller_step": seller_step,
            },
        )

    @staticmethod
    def uniform(
        n_each: int = 6,
        buyer_min: float = 60.0,
        buyer_max: float = 90.0,
        seller_min: float = 5.0,
        seller_max: float = 35.0,
        seed: Optional[int] = None,
    ) -> ValuationsSpec:
        """Uniform random valuations within specified ranges."""
        return ValuationsSpec(
            kind="uniform",
            n_buyers=n_each,
            n_sellers=n_each,
            params={
                "buyer_min": buyer_min,
                "buyer_max": buyer_max,
                "seller_min": seller_min,
                "seller_max": seller_max,
                "seed": seed,
            },
        )

    @staticmethod
    def fixed(
        buyer_valuations: list,
        seller_valuations: list,
    ) -> ValuationsSpec:
        """Explicit valuation lists for exact replication."""
        return ValuationsSpec(
            kind="fixed",
            n_buyers=len(buyer_valuations),
            n_sellers=len(seller_valuations),
            params={
                "buyer_valuations": list(buyer_valuations),
                "seller_valuations": list(seller_valuations),
            },
        )


class _Clearing:
    """Namespace for clearing-rule builders.

    Exposed as `Market.clearing`.
    """

    @staticmethod
    def uniform_price(tie_break: str = "random") -> ClearingSpec:
        """Single uniform clearing price for all matched pairs (paper Def. 1)."""
        return ClearingSpec(rule="uniform_price", tie_break=tie_break)

    @staticmethod
    def pairwise_midpoint(tie_break: str = "random") -> ClearingSpec:
        """Each matched pair clears at its own bid-ask midpoint."""
        return ClearingSpec(rule="pairwise_midpoint", tie_break=tie_break)


class Market:
    """Namespace for market-mechanism builders.

    The class itself holds no state. Use the staticmethods to construct
    immutable `MarketSpec` value objects.

    Nested namespaces:
      - `Market.valuations`: builders for `ValuationsSpec`
      - `Market.clearing`: builders for `ClearingSpec`
    """

    valuations = _Valuations
    clearing = _Clearing

    @staticmethod
    def call_auction(
        n_agents: int = 12,
        rounds: int = 20,
        valuations: Optional[ValuationsSpec] = None,
        clearing: Optional[ClearingSpec] = None,
        fee_bps: float = 0.0,
    ) -> MarketSpec:
        """A sealed-bid call auction."""
        if valuations is None:
            valuations = _Valuations.linear(n_each=n_agents // 2)
        if clearing is None:
            clearing = _Clearing.uniform_price()
        return MarketSpec(
            type="call_auction",
            n_agents=n_agents,
            rounds=rounds,
            valuations=valuations,
            clearing=clearing,
            fee_bps=fee_bps,
        )

    @staticmethod
    def cda(
        n_agents: int = 12,
        rounds: int = 20,
        valuations: Optional[ValuationsSpec] = None,
        clearing: Optional[ClearingSpec] = None,
        fee_bps: float = 0.0,
    ) -> MarketSpec:
        """A continuous double auction.

        Note. The current engine clears once per round; "continuous" here
        means pairwise-midpoint matching within the round, which is the
        empirically-used variant in the cognitive monoculture paper and one
        instance of the Lipschitz family of Proposition 3.
        """
        if valuations is None:
            valuations = _Valuations.linear(n_each=n_agents // 2)
        if clearing is None:
            clearing = _Clearing.pairwise_midpoint()
        return MarketSpec(
            type="cda",
            n_agents=n_agents,
            rounds=rounds,
            valuations=valuations,
            clearing=clearing,
            fee_bps=fee_bps,
        )
