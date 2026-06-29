"""Shared configuration primitives: Budget and Seed.

These types are deliberately small so they can be embedded inside any of
the larger spec types without circular imports. Both are immutable.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """A USD spending cap on a Study run.

    The cap is enforced at the call layer: before each LLM call, the runner
    estimates the call's cost; if `would_exceed(estimated_call_cost, spent)`
    returns True, `on_exceed` determines behavior.

    Attributes
    ----------
    usd
        Maximum total spend over the entire study run, summed across all
        replications.
    on_exceed
        What to do when a projected call would push spend over the cap:
        "halt" stops the study and raises a BudgetExceeded; "skip_replication"
        ends the current replication cleanly and continues; "warn" logs and
        proceeds.
    per_call_max_usd
        Optional upper bound on any single call's cost. Useful for catching
        pathological model failures that would otherwise burn through the
        budget.
    """

    usd: float = Field(gt=0)
    on_exceed: Literal["halt", "skip_replication", "warn"] = "halt"
    per_call_max_usd: Optional[float] = Field(default=None, gt=0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    def remaining(self, spent: float) -> float:
        return max(self.usd - spent, 0.0)

    def would_exceed(self, projected_call_cost: float, spent: float) -> bool:
        if self.per_call_max_usd is not None and projected_call_cost > self.per_call_max_usd:
            return True
        return (spent + projected_call_cost) > self.usd


class Seed(BaseModel):
    """A reproducibility seed.

    Studies derive per-replication seeds deterministically from `base` so
    that re-running a study with the same `base` reproduces the same trace.

    Attributes
    ----------
    base
        The integer seed. Per-replication seeds are `base + replication_index`.
    """

    base: int = 42

    model_config = ConfigDict(frozen=True, extra="forbid")

    def for_replication(self, replication_index: int) -> int:
        return self.base + replication_index


class BudgetExceeded(RuntimeError):
    """Raised when a Study run with `on_exceed="halt"` exceeds its Budget."""
