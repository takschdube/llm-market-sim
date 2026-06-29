"""Typed audit-schema-compliant trace of a study run.

`Trace` is the canonical data model produced by `Study.run`. It carries
the full study config, per-round results, per-decision data sufficient
for `Metric` and `Governance` analyses, and total cost accounting.

The same schema is the input contract for adapters that import traces
from external substrates (Concordia, ABIDES-MARL, Magentic Marketplace).
This is the audit schema referenced by the cognitive monoculture paper.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .metrics import MetricSpec


class TradeRecord(BaseModel):
    """One executed trade."""

    round: int
    buyer_id: str
    seller_id: str
    good: str
    price: float
    quantity: float

    model_config = ConfigDict(extra="forbid")


class Decision(BaseModel):
    """One agent decision in one round.

    The fields below define the audit schema v0.1. Substrate adapters
    that emit `Trace` objects must populate the required fields; optional
    fields default to safe values.
    """

    round: int
    agent_id: str
    role: Literal["buyer", "seller"]
    strategy_name: str
    provider: Optional[str] = None
    model: Optional[str] = None
    lineage_tag: Optional[str] = None
    prompt_hash: Optional[str] = None
    raw_response: Optional[str] = None
    submitted_price: float
    submitted_quantity: float = 1.0
    equilibrium_price: float
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    seed: int = 0
    timestamp: Optional[datetime] = None

    model_config = ConfigDict(extra="forbid")


class RoundResult(BaseModel):
    """All decisions and clearing outcomes for one round of one replication."""

    round: int
    mechanism: str
    clearing_price: Optional[float]
    n_trades: int = 0
    trades: List[TradeRecord] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Trace(BaseModel):
    """The audit-schema-compliant trace of a study run."""

    schema_version: Literal["0.1"] = "0.1"
    study_name: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    replications: List["Replication"] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(extra="forbid")

    @property
    def n_replications(self) -> int:
        return len(self.replications)

    @property
    def n_rounds(self) -> int:
        if not self.replications:
            return 0
        return sum(len(r.rounds) for r in self.replications)

    def all_decisions(self) -> List[Decision]:
        """Flatten all decisions across replications and rounds."""
        out: List[Decision] = []
        for rep in self.replications:
            for rnd in rep.rounds:
                out.extend(rnd.decisions)
        return out

    def evaluate(self, metrics: List["MetricSpec"]) -> Dict[str, Any]:
        """Compute the given metrics over this trace."""
        from .metrics import MetricRegistry

        results: Dict[str, Any] = {}
        for metric in metrics:
            fn = MetricRegistry.get(metric.name)
            results[metric.name] = fn(self, **metric.params)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class Replication(BaseModel):
    """One replication of a study run.

    A replication is a single (seed, market, population) execution. The
    `replication_index` is the zero-based index within the study.
    """

    replication_index: int
    seed: int
    rounds: List[RoundResult] = Field(default_factory=list)
    cost_usd: float = 0.0

    model_config = ConfigDict(extra="forbid")


# Resolve the forward reference now that Replication is defined.
Trace.model_rebuild()
