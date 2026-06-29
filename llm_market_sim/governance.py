"""Governance primitives: audit logs, deployment policies, compliance reports.

This is the niche-defining module. It treats a `Trace` as an audit
artifact, computes a deployment-relevant summary (`AuditLog`), and lets a
caller declare a `DeploymentPolicy` that the trace either satisfies or
fails. The output is a `ComplianceReport` suitable for handing to a
regulator or an internal review process.

The same `AuditLog` schema is the contract for traces imported from
external substrates via `Trace.from_concordia`, `Trace.from_abides`, and
similar adapters.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .populations import Population
from .traces import Trace


class AuditLog(BaseModel):
    """Deployment-relevant summary of a trace.

    Carries the architectural correlation rho_A with its confidence
    interval, the implied critical concentration c*, the realized LLM
    concentration in the trace, and the fleet lineage distribution.

    A regulator or internal reviewer scores this against a
    `DeploymentPolicy` to produce a `ComplianceReport`.
    """

    schema_version: str = "0.1"
    study_name: Optional[str] = None
    rho_a: float
    rho_a_ci: Tuple[float, float]
    critical_concentration: float
    realized_concentration: float
    fleet_lineage: Dict[str, float] = Field(default_factory=dict)
    fleet_provider: Dict[str, float] = Field(default_factory=dict)
    architectural_diversity_score: float = 0.0
    audit_timestamp: Optional[datetime] = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        realized_concentration: Optional[float] = None,
        fleet_lineage: Optional[Dict[str, float]] = None,
        fleet_provider: Optional[Dict[str, float]] = None,
        architectural_diversity_score: float = 0.0,
    ) -> "AuditLog":
        from src.analysis.icc import (
            critical_concentration as compute_c_star,
            estimate_within_model_icc,
        )

        from .metrics import _trace_to_icc_inputs

        inputs = _trace_to_icc_inputs(trace, "model")
        icc = estimate_within_model_icc(**inputs)
        n_agents = max(len(inputs["agent_models"]), 1)
        c_star = compute_c_star(icc.rho_hat, n_agents)

        if realized_concentration is None:
            llm_agents = sum(
                1
                for label in inputs["agent_models"].values()
                if not label.startswith("classical")
            )
            realized_concentration = (
                llm_agents / n_agents if n_agents else 0.0
            )

        return cls(
            study_name=trace.study_name,
            rho_a=icc.rho_hat if icc.rho_hat == icc.rho_hat else 0.0,
            rho_a_ci=icc.ci95,
            critical_concentration=c_star if c_star != float("inf") else 1.0,
            realized_concentration=realized_concentration,
            fleet_lineage=fleet_lineage or {},
            fleet_provider=fleet_provider or {},
            architectural_diversity_score=architectural_diversity_score,
            audit_timestamp=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class DeploymentPolicy(BaseModel):
    """A governance policy that a deployed multi-AI-agent market must satisfy.

    A policy is a set of thresholds. `check(audit)` produces a
    `ComplianceReport` that lists violations and recommendations.

    Attributes
    ----------
    concentration_limit
        Maximum allowed share of LLM agents in the deployed fleet.
        Typically derived from the critical concentration c* with a
        safety margin: `concentration_limit = c_lo - safety_margin`.
    rho_a_max
        Maximum acceptable architectural correlation. Above this, the
        fleet is considered too monocultural regardless of concentration.
    audit_schema_required
        If True, the trace's schema_version must match
        `audit_schema_version` for compliance.
    rotation_required
        If True, the fleet must rotate at least one provider/lineage
        within `rotation_period_days`.
    """

    concentration_limit: float = Field(ge=0.0, le=1.0)
    rho_a_max: float = Field(ge=0.0, le=1.0)
    audit_schema_required: bool = True
    audit_schema_version: str = "0.1"
    rotation_required: bool = False
    rotation_period_days: Optional[int] = None
    name: Optional[str] = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    def check(self, audit: AuditLog) -> "ComplianceReport":
        violations: List[str] = []
        recommendations: List[str] = []

        if self.audit_schema_required and audit.schema_version != self.audit_schema_version:
            violations.append(
                f"audit schema_version {audit.schema_version!r} does not match "
                f"required {self.audit_schema_version!r}"
            )

        if audit.realized_concentration > self.concentration_limit:
            violations.append(
                f"realized_concentration {audit.realized_concentration:.3f} "
                f"exceeds concentration_limit {self.concentration_limit:.3f}"
            )
            recommendations.append(
                f"reduce LLM share by approximately "
                f"{(audit.realized_concentration - self.concentration_limit) * 100:.1f}% "
                f"or rebalance with additional non-LLM agents"
            )

        if audit.rho_a > self.rho_a_max:
            violations.append(
                f"rho_A {audit.rho_a:.3f} exceeds rho_a_max {self.rho_a_max:.3f}"
            )
            recommendations.append(
                "diversify lineage: add at least one architecture from a "
                "different pretraining family"
            )

        if audit.critical_concentration < self.concentration_limit:
            recommendations.append(
                f"critical_concentration c*={audit.critical_concentration:.3f} is "
                f"below the policy's concentration_limit "
                f"{self.concentration_limit:.3f}; consider tightening the limit "
                f"to leave a safety margin"
            )

        if self.rotation_required and self.rotation_period_days is not None:
            recommendations.append(
                f"rotation policy: rotate at least one provider or lineage "
                f"within {self.rotation_period_days} days"
            )

        passed = len(violations) == 0
        return ComplianceReport(
            policy=self,
            audit=audit,
            passed=passed,
            violations=violations,
            recommendations=recommendations,
        )


class ComplianceReport(BaseModel):
    """The output of `DeploymentPolicy.check(audit)`."""

    policy: DeploymentPolicy
    audit: AuditLog
    passed: bool
    violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Compliance Report: {status}",
            "",
            f"- Study: {self.audit.study_name or 'unnamed'}",
            f"- rho_A: {self.audit.rho_a:.3f} "
            f"(95% CI {self.audit.rho_a_ci[0]:.3f}-{self.audit.rho_a_ci[1]:.3f})",
            f"- Critical concentration c*: {self.audit.critical_concentration:.3f}",
            f"- Realized concentration: {self.audit.realized_concentration:.3f}",
            f"- Architectural diversity: {self.audit.architectural_diversity_score:.3f}",
            "",
            f"## Policy: {self.policy.name or 'unnamed'}",
            f"- concentration_limit: {self.policy.concentration_limit:.3f}",
            f"- rho_a_max: {self.policy.rho_a_max:.3f}",
        ]
        if self.violations:
            lines.append("")
            lines.append("## Violations")
            for v in self.violations:
                lines.append(f"- {v}")
        if self.recommendations:
            lines.append("")
            lines.append("## Recommendations")
            for r in self.recommendations:
                lines.append(f"- {r}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class FleetDiversity(BaseModel):
    """Architectural-diversity analysis of a deployed population."""

    population: Population
    lineage_distribution: Dict[str, float]
    provider_distribution: Dict[str, float]
    architectural_diversity_score: float
    llm_concentration: float

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @classmethod
    def from_population(cls, population: Population) -> "FleetDiversity":
        return cls(
            population=population,
            lineage_distribution=population.lineage_distribution(),
            provider_distribution=population.provider_distribution(),
            architectural_diversity_score=population.architectural_diversity_score(),
            llm_concentration=population.llm_concentration(),
        )

    def rho_a_implied(self, baseline_rho: float = 0.9) -> float:
        """Naive prediction: rho_A * (1 - diversity_score).

        This is a fast estimate, not the rigorous derived quantity. Use
        `Metric.icc_within` on a trace for the rigorous estimate.
        """
        return float(baseline_rho * (1.0 - self.architectural_diversity_score))


class Governance:
    """Namespace for governance accessors."""

    @staticmethod
    def audit(
        trace: Trace,
        population: Optional[Population] = None,
    ) -> AuditLog:
        """Build an AuditLog from a Trace.

        If `population` is supplied, fleet-level metadata (lineage
        distribution, diversity score) is filled in. Otherwise the
        AuditLog carries only the trace-derived rho_A and c*.
        """
        fleet_lineage: Dict[str, float] = {}
        fleet_provider: Dict[str, float] = {}
        architectural_diversity = 0.0
        realized_concentration: Optional[float] = None
        if population is not None:
            fleet_lineage = population.lineage_distribution()
            fleet_provider = population.provider_distribution()
            architectural_diversity = population.architectural_diversity_score()
            realized_concentration = population.llm_concentration()

        return AuditLog.from_trace(
            trace,
            realized_concentration=realized_concentration,
            fleet_lineage=fleet_lineage,
            fleet_provider=fleet_provider,
            architectural_diversity_score=architectural_diversity,
        )

    @staticmethod
    def policy(
        concentration_limit: float,
        rho_a_max: float,
        audit_schema_required: bool = True,
        audit_schema_version: str = "0.1",
        rotation_required: bool = False,
        rotation_period_days: Optional[int] = None,
        name: Optional[str] = None,
    ) -> DeploymentPolicy:
        return DeploymentPolicy(
            concentration_limit=concentration_limit,
            rho_a_max=rho_a_max,
            audit_schema_required=audit_schema_required,
            audit_schema_version=audit_schema_version,
            rotation_required=rotation_required,
            rotation_period_days=rotation_period_days,
            name=name,
        )

    @staticmethod
    def fleet(population: Population) -> FleetDiversity:
        return FleetDiversity.from_population(population)
