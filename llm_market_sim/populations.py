"""Heterogeneous populations of agents.

A `Population` is an ordered tuple of `AgentSpec`s. Order is meaningful
for reproducibility: agents are expanded by walking the population in
order and assigning roles to satisfy the market's buyer/seller split.

The `lineage_distribution` and `architectural_diversity_score` properties
feed `Governance.fleet`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .agents import AgentSpec


class Population(BaseModel):
    """An ordered collection of `AgentSpec`s.

    Two populations with the same `agents` field in the same order produce
    identical results given the same seed.
    """

    agents: List[AgentSpec] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @classmethod
    def of(cls, *specs: AgentSpec) -> "Population":
        """Build a Population from a sequence of `AgentSpec`s."""
        return cls(agents=list(specs))

    @property
    def total_agents(self) -> int:
        return sum(spec.count for spec in self.agents)

    def lineage_distribution(self) -> Dict[str, float]:
        """Return the share of the population by `lineage_tag`.

        Agents without a lineage_tag are bucketed as "unknown".
        """
        if self.total_agents == 0:
            return {}
        counts: Dict[str, int] = {}
        for spec in self.agents:
            tag = spec.lineage_tag or "unknown"
            counts[tag] = counts.get(tag, 0) + spec.count
        n = self.total_agents
        return {tag: c / n for tag, c in counts.items()}

    def provider_distribution(self) -> Dict[str, float]:
        """Return the share of the population by LLM provider.

        Classical and custom agents are bucketed as "classical" and
        "custom" respectively.
        """
        if self.total_agents == 0:
            return {}
        counts: Dict[str, int] = {}
        for spec in self.agents:
            key = spec.provider if spec.kind == "llm" else spec.kind
            key = key or "unknown"
            counts[key] = counts.get(key, 0) + spec.count
        n = self.total_agents
        return {k: c / n for k, c in counts.items()}

    def architectural_diversity_score(self) -> float:
        """Effective number of architectures, normalized to [0, 1].

        Computed as 1 - H(p)_norm where p is the lineage distribution and
        H is the normalized Shannon entropy. A monoculture (one lineage)
        returns 0; a uniform mix across k lineages returns 1 - 1/log(k).
        """
        import math

        dist = self.lineage_distribution()
        if not dist or len(dist) == 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in dist.values() if p > 0)
        max_entropy = math.log(len(dist))
        if max_entropy == 0:
            return 0.0
        # Higher entropy means more diverse; normalize and return.
        return float(entropy / max_entropy)

    def llm_concentration(self) -> float:
        """Share of the population that is LLM-backed (regardless of provider)."""
        if self.total_agents == 0:
            return 0.0
        llm_count = sum(spec.count for spec in self.agents if spec.kind == "llm")
        return llm_count / self.total_agents

    def _expand_to_engine_agents(self, profiles: Sequence[Any]) -> List[Any]:
        """Expand the population into concrete engine agents.

        The runner calls this with a precomputed list of profiles
        (buyer profiles first, then seller profiles). The population is
        walked in order; each AgentSpec consumes `spec.count` profiles.
        Buyer specs and seller specs are interleaved by the runner via
        valuation-scheme ordering.
        """
        if len(profiles) != self.total_agents:
            raise ValueError(
                f"Population has {self.total_agents} agents but {len(profiles)} "
                f"profiles were supplied"
            )

        engine_agents: List[Any] = []
        cursor = 0
        for spec in self.agents:
            sub = profiles[cursor : cursor + spec.count]
            engine_agents.extend(spec._to_engine_agents(sub))
            cursor += spec.count
        return engine_agents
