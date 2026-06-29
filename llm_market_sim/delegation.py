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

from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Sequence, Tuple

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


def probe_proxy(proxy: Callable[[Action], Action],
                reports: Iterable[Action],
                utility: Utility,
                *,
                guardrail: Optional[Callable[[Action], Action]] = None,
                honest_report: Optional[Action] = None) -> Dict[str, Any]:
    """Sweep a proxy through a grid of reports and measure the incentive to misreport.

    This is the empirical counterpart of :func:`within_range_regret`, and it works
    for any report-to-action map: an analytic proxy (a bid-scaling rule) or a
    language-model proxy (a callable that prompts an LLM with the report and parses
    an action). It makes one ``proxy`` call per report, so an LLM proxy costs
    ``len(reports)`` model calls.

    ``proxy`` maps a report to the action the mechanism sees (the principal's only
    lever is the report). ``reports`` is the grid of self-descriptions to sweep.
    ``utility`` maps an action to the principal's interim utility with opponents
    fixed. ``guardrail`` is an optional action-to-action map composed onto the proxy
    output (a cap, a filter, an alignment layer). ``honest_report`` is the truthful
    report; it defaults to the middle of ``reports``.

    Returns a dict with ``reachable`` (the (report, action) pairs reached),
    ``honest_action``, ``within_range_regret`` (best reachable utility minus the
    honest action's, the gain from misreporting by the truth-to-proxy identity),
    ``best_report``, ``best_action``, and ``inflation`` = ``best_report /
    honest_report``, the input/prompt inflation the principal needs to recover the
    best reachable action.
    """
    c = guardrail if guardrail is not None else (lambda a: a)
    reports = list(reports)
    if not reports:
        raise ValueError("reports must be non-empty")
    if honest_report is None:
        honest_report = reports[len(reports) // 2]
    reached = [(r, c(proxy(r))) for r in reports]
    honest_action = c(proxy(honest_report))
    best_report, best_action = max(reached, key=lambda ra: utility(ra[1]))
    within = utility(best_action) - utility(honest_action)
    inflation = (best_report / honest_report) if honest_report else float("nan")
    return {
        "reachable": reached,
        "honest_action": honest_action,
        "best_report": best_report,
        "best_action": best_action,
        "within_range_regret": float(within),
        "inflation": float(inflation),
    }


if __name__ == "__main__":
    # Self-check: an analytic bid-scaling proxy under a soft-cap guardrail.
    # pi(report) = report / 2; soft cap compresses bids above kappa = 0.3.
    # True value v = 1.5, so the unconstrained best bid v/2 = 0.75 > kappa: the
    # guardrail binds, leaving positive within-range regret and inflation > 1.
    kappa, slope = 0.30, 0.4
    soft_cap = lambda b: b if b <= kappa else kappa + slope * (b - kappa)
    v = 1.5
    util = lambda bid: v * bid - bid * bid          # peak at bid = v/2 = 0.75
    out = probe_proxy(lambda r: r / 2.0,
                      reports=[i / 100.0 for i in range(5, 400)],
                      utility=util,
                      guardrail=soft_cap,
                      honest_report=v)
    assert out["within_range_regret"] > 0, out
    assert out["inflation"] > 1.0, out
    print("probe_proxy self-check OK:",
          {k: round(out[k], 3) for k in ("within_range_regret", "inflation", "best_report")})
