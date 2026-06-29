"""Engine adapter: translates public specs into engine objects and runs them.

Internal module. The public API in `llm_market_sim/` constructs Pydantic
spec objects; this module is the one place that knows how to translate
those specs into the existing engine in `src/` and how to capture the
engine's outputs into the public `Trace` schema.

Keeping this isolated has two benefits. First, the public surface stays
clean: users never see the engine. Second, the engine can evolve
internally without breaking the public API; only this file needs to
adjust.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from .studies import Study

from .traces import Decision, Replication, RoundResult, TradeRecord, Trace


def run_study_replications(
    study: "Study", parallel: int = 1, resume: bool = False
) -> Trace:
    """Execute a Study across its replications and produce a Trace."""
    from src.simulation.runner import Simulation

    started_at = datetime.now(timezone.utc)
    valuations_scheme = study.market.valuations.to_engine()
    mechanism_name = study.market.clearing.engine_mechanism_name

    replications: List[Replication] = []
    total_cost = 0.0

    for replication_index in range(study.replications):
        seed = study.seed.for_replication(replication_index)
        random.seed(seed)

        profiles = valuations_scheme.generate_profiles(
            study.market.valuations.n_buyers,
            study.market.valuations.n_sellers,
        )
        engine_agents = study.population._expand_to_engine_agents(profiles)

        sim = Simulation(
            agents=engine_agents,
            num_rounds=study.market.rounds,
            parallel=parallel > 1,
            mechanism=mechanism_name,
            memoryless=study.memoryless,
        )
        sim.run()

        # Compute equilibrium price from valuations (Walrasian midpoint).
        equilibrium_price = _equilibrium_price(profiles)

        rounds: List[RoundResult] = []
        for round_idx, round_info in enumerate(sim.round_data, start=1):
            decisions = _collect_decisions(
                engine_agents=engine_agents,
                round_num=round_idx,
                equilibrium_price=equilibrium_price,
                seed=seed,
                study_population=study.population,
            )
            trades = [
                TradeRecord(
                    round=round_idx,
                    buyer_id=t.buyer_id,
                    seller_id=t.seller_id,
                    good=t.good,
                    price=t.price,
                    quantity=t.quantity,
                )
                for t in round_info.get("trades", [])
            ]
            clearing_price = (
                round_info["prices"][0]
                if round_info.get("prices")
                else None
            )
            rounds.append(
                RoundResult(
                    round=round_idx,
                    mechanism=mechanism_name,
                    clearing_price=clearing_price,
                    n_trades=round_info.get("num_trades", 0),
                    trades=trades,
                    decisions=decisions,
                )
            )

        replication_cost = sum(d.cost_usd for r in rounds for d in r.decisions)
        total_cost += replication_cost

        replications.append(
            Replication(
                replication_index=replication_index,
                seed=seed,
                rounds=rounds,
                cost_usd=replication_cost,
            )
        )

    completed_at = datetime.now(timezone.utc)
    return Trace(
        study_name=study.name,
        config=study.model_dump(mode="json"),
        replications=replications,
        total_cost_usd=total_cost,
        started_at=started_at,
        completed_at=completed_at,
    )


def _equilibrium_price(profiles: List[Any]) -> float:
    """Walrasian midpoint of the marginal buyer and seller valuations."""
    buyer_vals = sorted(
        [p.valuation for p in profiles if p.role == "buyer"], reverse=True
    )
    seller_vals = sorted([p.valuation for p in profiles if p.role == "seller"])
    if not buyer_vals or not seller_vals:
        return 0.0
    # Pair off until exchange ceases to be efficient.
    k = 0
    while k < min(len(buyer_vals), len(seller_vals)) and buyer_vals[k] > seller_vals[k]:
        k += 1
    if k == 0:
        return 0.0
    return (buyer_vals[k - 1] + seller_vals[k - 1]) / 2.0


def _collect_decisions(
    engine_agents: List[Any],
    round_num: int,
    equilibrium_price: float,
    seed: int,
    study_population: Any,
) -> List[Decision]:
    """Extract this round's per-agent decisions from the engine logs.

    Returns a list of `Decision` objects, one per agent per round.
    """
    decisions: List[Decision] = []
    # Build a lookup of agent_id -> (provider, model, lineage_tag, strategy_name)
    # by walking the population specs in order against the engine agents.
    metadata = _agent_metadata_map(engine_agents, study_population)

    for agent in engine_agents:
        agent_logs = [
            log for log in agent.decision_logs if log.round == round_num
        ]
        if not agent_logs:
            continue
        log = agent_logs[-1]
        meta = metadata.get(agent.state.id, {})
        prompt = log.observation or ""
        prompt_hash = (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            if prompt
            else None
        )
        decisions.append(
            Decision(
                round=round_num,
                agent_id=agent.state.id,
                role=agent.state.role,
                strategy_name=meta.get("strategy_name", log.agent_type),
                provider=meta.get("provider") or log.provider,
                model=meta.get("model") or log.model,
                lineage_tag=meta.get("lineage_tag"),
                prompt_hash=prompt_hash,
                raw_response=log.raw_response or None,
                submitted_price=(
                    log.order.price
                    if log.order is not None
                    else log.decision.get("price", 0.0)
                ),
                submitted_quantity=(
                    log.order.quantity
                    if log.order is not None
                    else log.decision.get("quantity", 1.0)
                ),
                equilibrium_price=equilibrium_price,
                cost_usd=0.0,
                tokens_in=getattr(log, "tokens_used", 0),
                tokens_out=0,
                latency_ms=getattr(log, "latency_ms", 0.0),
                seed=seed,
                timestamp=datetime.now(timezone.utc),
            )
        )
    return decisions


def _agent_metadata_map(
    engine_agents: List[Any], population: Any
) -> Dict[str, Dict[str, Any]]:
    """Reconstruct (provider, model, lineage, strategy_name) per agent."""
    metadata: Dict[str, Dict[str, Any]] = {}
    cursor = 0
    for spec in population.agents:
        for _ in range(spec.count):
            if cursor >= len(engine_agents):
                break
            agent_id = engine_agents[cursor].state.id
            metadata[agent_id] = {
                "provider": spec.provider,
                "model": spec.model,
                "lineage_tag": spec.lineage_tag,
                "strategy_name": spec.strategy.name,
            }
            cursor += 1
    return metadata
