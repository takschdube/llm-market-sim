"""Delegated-play analysis: within-range regret and the truth-to-proxy identity.

Reference implementation of the primitives in "When Is Delegated Play Truthful?
Within-Range Regret and the Limits of Aligned Proxies" (Dube, 2026). A *proxy* is a
map from a principal's report to an action in a mechanism; an autobidder and a
language-model agent are both proxies. The proxy's *within-range regret* is its
regret restricted to the actions it can reach, and it equals the principal's gain
from misreporting to its own proxy (the truth-to-proxy identity, Theorem 1 of the
paper). These functions are the reference implementation used for the paper's
numerical validation; they are deliberately mechanism-agnostic, taking a utility
callable and a set of reachable actions.

    from llm_market_sim.delegation import (
        within_range_regret, manipulation_gain, guardrail_metrics,
        recertification_count,
    )
"""
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Sequence, Tuple

Action = Hashable
Utility = Callable[[Action], float]


def within_range_regret(utility: Utility,
                        reachable: Iterable[Action],
                        truthful_action: Action) -> float:
    """W_i: the best reachable action's utility minus the truthful action's utility.

    ``utility`` is the principal's interim utility as a function of its proxy's
    action (opponents held fixed); ``reachable`` is the proxy's reachable set
    Reach_i; ``truthful_action`` is the action the proxy plays under honest reporting.
    """
    best = max(utility(a) for a in reachable)
    return best - utility(truthful_action)


def manipulation_gain(utility: Utility,
                      proxy: Callable[[Action], Action],
                      report_space: Iterable[Action],
                      true_type: Action) -> float:
    """G_i: the most a principal gains by misreporting its type to its own proxy.

    Equal to :func:`within_range_regret` over the proxy's reachable set by the
    truth-to-proxy identity. Computed here through the report channel (an
    independent code path) so a test can confirm G == W rather than assume it.
    """
    truthful = utility(proxy(true_type))
    best = max(utility(proxy(t)) for t in report_space)
    return best - truthful


def is_range_optimal(utility: Utility, reachable: Iterable[Action],
                     truthful_action: Action, tol: float = 1e-9) -> bool:
    """Whether honest reporting is optimal at this type (W_i = 0)."""
    return within_range_regret(utility, reachable, truthful_action) <= tol


def guardrail_metrics(utility: Utility,
                      constrained_reach: Iterable[Action],
                      honest_optimum: Action,
                      guardrailed_truthful_action: Action) -> Tuple[float, float]:
    """The two quantities behind the trilemma (Theorem 2), for a guardrailed proxy.

    Returns ``(within_range_regret, capability_gap)``. ``honest_optimum`` is the
    best action over the *unconstrained* reachable set; ``constrained_reach`` is the
    set the guardrail leaves reachable; ``guardrailed_truthful_action`` is what
    honest reporting produces after the guardrail. The guardrail is *truthful* iff
    the within-range regret is zero and *capability-preserving* iff the capability
    gap is zero; *binding* (the third property) is a statement about the action the
    caller supplies, so the caller classifies it.
    """
    best_reachable = max(utility(a) for a in constrained_reach)
    within = best_reachable - utility(guardrailed_truthful_action)
    gap = utility(honest_optimum) - best_reachable
    return within, gap


def recertification_count(drifts: Sequence[float], lipschitz: float,
                          n_agents: int, slack: float) -> int:
    """Number of re-certifications a drifting proxy stream triggers (online re-cert).

    Re-certify whenever the manipulation-gain drift accumulated since the last
    certificate, bounded by ``2 * lipschitz * n_agents * drift`` per update (the
    drift lemma), would exceed ``slack``. The count is at most
    ``ceil(2 * lipschitz * n_agents * sum(drifts) / slack) + 1``, so it scales with
    total drift rather than the number of updates.
    """
    acc, count = 0.0, 0
    for d in drifts:
        acc += 2.0 * lipschitz * n_agents * d
        if acc > slack:
            count += 1
            acc = 0.0
    return count
