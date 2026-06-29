"""Study specification and the StudyRunner that executes it.

A `Study` is a declarative specification of an experiment: a market, a
population, a number of replications, a seed, and an optional budget.
`Study.run` constructs a `Trace`. The runner delegates to the existing
engine `Simulation` for the actual round-by-round execution and captures
its outputs into the `Trace` schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .config import Budget, Seed
from .markets import MarketSpec
from .populations import Population

if TYPE_CHECKING:
    from .metrics import MetricSpec
    from .traces import Decision, Replication, RoundResult, TradeRecord, Trace


class Study(BaseModel):
    """A reproducible experiment specification.

    Constructing a `Study` does not run anything. Call `.run()` to execute
    and obtain a `Trace`.

    `memoryless` is a study-level flag: when True, agents see no transcript
    of prior rounds and their internal history is cleared at the start of
    every round. Decision logs are preserved on the agent for analysis. The
    flag is the experimental protocol that tests the ICC mechanism of
    Theorem 1 in the cognitive monoculture paper.
    """

    name: Optional[str] = None
    market: MarketSpec
    population: Population
    replications: int = Field(default=30, ge=1)
    seed: Seed = Field(default_factory=Seed)
    budget: Optional[Budget] = None
    cache_dir: Optional[str] = None
    memoryless: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def run(self, parallel: int = 1, resume: bool = False) -> "Trace":
        """Execute the study and return an audit-schema-compliant Trace.

        `parallel` controls intra-replication parallelism for LLM calls
        (the engine `Simulation` runs agent decisions concurrently when
        the value is > 1 and any LLM agents are present). Replications
        run sequentially in this version; multi-replication parallelism
        is on the roadmap and will require process-level isolation
        because the engine's RNG is module-level.
        """
        from ._engine_adapter import run_study_replications

        return run_study_replications(self, parallel=parallel, resume=resume)

    def with_memoryless(self, memoryless: bool = True) -> "Study":
        return self.model_copy(update={"memoryless": memoryless})

    # Fluent mutators returning a new (frozen) Study with one field changed.

    def with_market(self, market: MarketSpec) -> "Study":
        return self.model_copy(update={"market": market})

    def with_population(self, population: Population) -> "Study":
        return self.model_copy(update={"population": population})

    def with_replications(self, n: int) -> "Study":
        return self.model_copy(update={"replications": n})

    def with_budget(self, budget: Budget) -> "Study":
        return self.model_copy(update={"budget": budget})

    def with_seed(self, seed: Seed) -> "Study":
        return self.model_copy(update={"seed": seed})

    # Static comparison helper.

    @staticmethod
    def compare(
        studies: List["Study"], metrics: List[Union[str, "MetricSpec"]]
    ) -> "Comparison":
        """Run each study and produce a side-by-side Comparison."""
        from .metrics import Metric, MetricSpec

        traces = [s.run() for s in studies]
        metric_specs: List[MetricSpec] = []
        for m in metrics:
            if isinstance(m, str):
                metric_specs.append(MetricSpec(name=m, params={}))
            else:
                metric_specs.append(m)

        rows: List[Dict[str, Any]] = []
        for study, trace in zip(studies, traces):
            row: Dict[str, Any] = {"study": study.name or "(unnamed)"}
            evaluated = trace.evaluate(metric_specs)
            row.update(evaluated)
            rows.append(row)
        return Comparison(rows=rows, studies=list(studies), traces=traces)


class Comparison(BaseModel):
    """Side-by-side comparison of Study runs."""

    rows: List[Dict[str, Any]]
    studies: List[Study]
    traces: List["Trace"]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def to_markdown(self) -> str:
        return self.to_dataframe().to_markdown(index=False)


# Resolve forward refs at module load.
def _resolve():
    from .traces import Trace  # noqa: F401

    Comparison.model_rebuild()


_resolve()
