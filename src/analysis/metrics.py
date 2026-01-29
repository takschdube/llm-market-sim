# src/analysis/metrics.py
"""
Market Metrics
==============
Functions for computing market efficiency, convergence, and behavioral metrics.

Core metrics:
- compute_walrasian_price: Theoretical equilibrium price
- compute_convergence: Mean absolute deviation from equilibrium
- compute_efficiency: Allocative efficiency (realized/max surplus)

Behavioral metrics (for cognitive monoculture analysis):
- compute_price_entropy: Shannon entropy of price distribution
- compute_bid_ask_entropy: Entropy of bid/ask distributions per round
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.agents import BaseAgent


def compute_efficiency(agents: List[BaseAgent], trades: List, good: str = "good_A") -> float:
    """
    Realized gains / Maximum possible gains (allocative efficiency).

    For each trade, the surplus is:
    buyer_valuation - seller_valuation (price cancels out)
    """
    if not trades:
        return 0.0

    # Build lookup for agent valuations
    agent_vals = {a.state.id: a.state.valuation.get(good, 0) for a in agents}

    realized_surplus = 0.0
    for t in trades:
        buyer_val = agent_vals.get(t.buyer_id, 0)
        seller_val = agent_vals.get(t.seller_id, 0)
        # Surplus = buyer's gain + seller's gain = (buyer_val - price) + (price - seller_val)
        realized_surplus += (buyer_val - seller_val) * t.quantity

    max_surplus = compute_max_surplus(agents, good)
    return realized_surplus / max_surplus if max_surplus > 0 else 0.0


def compute_max_surplus(agents: List[BaseAgent], good: str = "good_A") -> float:
    """
    Maximum possible surplus from efficient allocation.

    Match highest-valuation buyers with lowest-valuation sellers
    until no more gains from trade exist.
    """
    # Separate buyers (have money, want goods) and sellers (have goods)
    buyers = []
    sellers = []

    for a in agents:
        val = a.state.valuation.get(good, 0)
        goods_held = a.state.endowment.get(good, 0)
        money_held = a.state.endowment.get("money", 0)

        if goods_held > 0:
            # Seller: can sell up to goods_held units at their valuation
            sellers.append((val, goods_held))
        if money_held > 0:
            # Buyer: willing to buy at their valuation
            # Max units they can buy depends on price, but for max surplus we just need valuations
            buyers.append((val, money_held))

    # Sort: highest valuation buyers first, lowest valuation sellers first
    buyers.sort(key=lambda x: -x[0])  # descending
    sellers.sort(key=lambda x: x[0])   # ascending

    max_surplus = 0.0
    buyer_idx = 0
    seller_idx = 0

    while buyer_idx < len(buyers) and seller_idx < len(sellers):
        buyer_val, _ = buyers[buyer_idx]
        seller_val, _ = sellers[seller_idx]

        # Trade only if buyer values more than seller
        if buyer_val <= seller_val:
            break

        # Each efficient trade creates surplus = buyer_val - seller_val
        # For simplicity, assume 1 unit trades at a time
        max_surplus += buyer_val - seller_val
        buyer_idx += 1
        seller_idx += 1

    return max_surplus


def compute_convergence(prices: List[float], equilibrium_price: float) -> float:
    """Mean absolute deviation from equilibrium price."""
    if not prices:
        return float("inf")
    return sum(abs(p - equilibrium_price) for p in prices) / len(prices)


def compute_walrasian_price(agents: List[BaseAgent], good: str = "good_A") -> float:
    """
    Theoretical Walrasian equilibrium price where supply = demand.

    Build demand curve (buyer valuations, sorted descending)
    Build supply curve (seller valuations, sorted ascending)
    Equilibrium is where they intersect.
    """
    # Collect buyer valuations (demand) and seller valuations (supply)
    demand = []  # (valuation, quantity)
    supply = []  # (valuation, quantity)

    for a in agents:
        val = a.state.valuation.get(good, 0)
        goods_held = a.state.endowment.get(good, 0)
        money_held = a.state.endowment.get("money", 0)

        if goods_held > 0:
            # Seller willing to sell at their valuation or higher
            supply.append(val)
        if money_held > 0:
            # Buyer willing to buy at their valuation or lower
            demand.append(val)

    if not demand or not supply:
        return 0.0

    # Sort demand descending (highest willingness to pay first)
    demand.sort(reverse=True)
    # Sort supply ascending (lowest reservation price first)
    supply.sort()

    # Find intersection: where marginal buyer valuation >= marginal seller valuation
    # The equilibrium price is typically the midpoint at the intersection
    equilibrium_price = 0.0
    for i in range(min(len(demand), len(supply))):
        if demand[i] >= supply[i]:
            # This trade can happen; equilibrium is between these valuations
            equilibrium_price = (demand[i] + supply[i]) / 2
        else:
            break

    return equilibrium_price


def compute_price_entropy(prices: List[float], n_bins: int = 10,
                          price_range: Optional[Tuple[float, float]] = None) -> float:
    """
    Compute Shannon entropy of price distribution.

    Lower entropy indicates prices are concentrating (convergence).
    Higher entropy indicates prices are spread out (no convergence).

    This metric is useful for detecting cognitive monoculture:
    - ZI agents: entropy should stay flat over rounds (no learning)
    - LLM agents: entropy should decrease over rounds (convergence)
    - Mixed LLM models: entropy may decrease slower than homogeneous

    Args:
        prices: List of transaction prices
        n_bins: Number of bins for histogram (default 10)
        price_range: (min, max) range for binning. If None, uses data range.

    Returns:
        Shannon entropy in bits. Higher = more spread out.
    """
    if not prices or len(prices) < 2:
        return 0.0

    prices_arr = np.array(prices)

    if price_range is None:
        price_range = (prices_arr.min(), prices_arr.max())

    # Handle case where all prices are the same
    if price_range[0] == price_range[1]:
        return 0.0

    # Create histogram
    hist, _ = np.histogram(prices_arr, bins=n_bins, range=price_range)

    # Convert to probabilities
    total = hist.sum()
    if total == 0:
        return 0.0

    probs = hist / total

    # Compute Shannon entropy: -sum(p * log2(p)) for p > 0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_bid_ask_entropy(bids: List[float], asks: List[float],
                            n_bins: int = 10,
                            price_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """
    Compute Shannon entropy of bid and ask distributions separately.

    Useful for understanding whether buyers and sellers are converging
    differently. In cognitive monoculture, we expect:
    - Same-model markets: both bid and ask entropy decrease together
    - Mixed-model markets: entropy decreases more slowly

    Args:
        bids: List of bid prices this round
        asks: List of ask prices this round
        n_bins: Number of bins for histogram
        price_range: (min, max) range for binning

    Returns:
        Dict with 'bid_entropy', 'ask_entropy', 'combined_entropy'
    """
    # Determine price range from all prices if not provided
    if price_range is None and (bids or asks):
        all_prices = bids + asks
        if all_prices:
            price_range = (min(all_prices), max(all_prices))

    bid_entropy = compute_price_entropy(bids, n_bins, price_range) if bids else 0.0
    ask_entropy = compute_price_entropy(asks, n_bins, price_range) if asks else 0.0

    # Combined entropy treats all prices as one distribution
    combined = bids + asks
    combined_entropy = compute_price_entropy(combined, n_bins, price_range) if combined else 0.0

    return {
        "bid_entropy": bid_entropy,
        "ask_entropy": ask_entropy,
        "combined_entropy": combined_entropy,
    }


def compute_entropy_trajectory(round_data: List[Dict],
                               n_bins: int = 10) -> List[Dict]:
    """
    Compute entropy metrics for each round in a simulation.

    This creates a trajectory of entropy over time, useful for visualizing
    convergence patterns.

    Args:
        round_data: List of round results from Simulation.run()
        n_bins: Number of bins for entropy calculation

    Returns:
        List of dicts with round number and entropy metrics
    """
    if not round_data:
        return []

    # Determine global price range for consistent binning
    all_prices = []
    for r in round_data:
        all_prices.extend(r.get("prices", []))

    if not all_prices:
        return []

    price_range = (min(all_prices), max(all_prices))

    trajectory = []
    for r in round_data:
        prices = r.get("prices", [])
        entropy = compute_price_entropy(prices, n_bins, price_range)

        trajectory.append({
            "round": r["round"],
            "price_entropy": entropy,
            "num_trades": r.get("num_trades", 0),
            "mean_price": np.mean(prices) if prices else None,
            "std_price": np.std(prices) if len(prices) > 1 else 0.0,
        })

    return trajectory


def compute_action_correlation(decisions_a: List[Dict], decisions_b: List[Dict],
                               metric: str = "price") -> float:
    """
    Compute correlation between two agents' decisions over rounds.

    This is the key metric for cognitive monoculture analysis:
    - Same-model agents should show HIGH correlation
    - Different-model agents should show LOWER correlation

    Args:
        decisions_a: List of decision dicts from agent A
        decisions_b: List of decision dicts from agent B
        metric: Which field to correlate ("price", "action")

    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    if not decisions_a or not decisions_b:
        return 0.0

    # Match by round
    a_by_round = {d.get("round"): d for d in decisions_a}
    b_by_round = {d.get("round"): d for d in decisions_b}

    common_rounds = set(a_by_round.keys()) & set(b_by_round.keys())

    if len(common_rounds) < 3:
        return 0.0

    values_a = []
    values_b = []

    for round_num in sorted(common_rounds):
        da = a_by_round[round_num]
        db = b_by_round[round_num]

        if metric == "price":
            # Get price from decision dict
            price_a = da.get("decision", {}).get("price")
            price_b = db.get("decision", {}).get("price")

            if price_a is not None and price_b is not None:
                values_a.append(price_a)
                values_b.append(price_b)

        elif metric == "action":
            # Encode action as numeric: buy=1, sell=-1, hold=0
            action_map = {"buy": 1, "sell": -1, "hold": 0}
            action_a = da.get("decision", {}).get("action", "hold")
            action_b = db.get("decision", {}).get("action", "hold")
            values_a.append(action_map.get(action_a, 0))
            values_b.append(action_map.get(action_b, 0))

    if len(values_a) < 3:
        return 0.0

    # Compute Pearson correlation
    return float(np.corrcoef(values_a, values_b)[0, 1])


def compute_model_correlation_matrix(logs_by_agent: Dict[str, List[Dict]],
                                     agent_models: Dict[str, str]) -> Dict:
    """
    Compute correlation matrix between agents, grouped by model.

    This is the key analysis for cognitive monoculture:
    - Within-model correlations (same model pairs)
    - Between-model correlations (different model pairs)

    If cognitive monoculture exists, within > between.

    Args:
        logs_by_agent: Dict mapping agent_id to list of decision logs
        agent_models: Dict mapping agent_id to model name

    Returns:
        Dict with:
        - 'within_model_corr': Mean correlation for same-model pairs
        - 'between_model_corr': Mean correlation for different-model pairs
        - 'correlation_matrix': Full pairwise correlation matrix
        - 'model_groups': Which agents belong to which model
    """
    agent_ids = list(logs_by_agent.keys())

    if len(agent_ids) < 2:
        return {
            "within_model_corr": 0.0,
            "between_model_corr": 0.0,
            "correlation_matrix": {},
            "model_groups": {},
        }

    # Compute pairwise correlations
    correlations = {}
    within_corrs = []
    between_corrs = []

    for i, agent_a in enumerate(agent_ids):
        for agent_b in agent_ids[i + 1:]:
            corr = compute_action_correlation(
                logs_by_agent[agent_a],
                logs_by_agent[agent_b],
                metric="price"
            )

            pair_key = f"{agent_a}_{agent_b}"
            correlations[pair_key] = corr

            # Classify as within or between model
            model_a = agent_models.get(agent_a, "unknown")
            model_b = agent_models.get(agent_b, "unknown")

            if model_a == model_b:
                within_corrs.append(corr)
            else:
                between_corrs.append(corr)

    # Group agents by model
    model_groups = {}
    for agent_id, model in agent_models.items():
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(agent_id)

    return {
        "within_model_corr": np.mean(within_corrs) if within_corrs else 0.0,
        "between_model_corr": np.mean(between_corrs) if between_corrs else 0.0,
        "within_model_corrs": within_corrs,
        "between_model_corrs": between_corrs,
        "correlation_matrix": correlations,
        "model_groups": model_groups,
    }