#!/usr/bin/env python
"""
View graphs from saved simulation results.

Usage:
    # View the most recent simulation
    uv run python -m src.analysis.view_results

    # View a specific simulation by ID
    uv run python -m src.analysis.view_results 20260101006

    # List all available simulations
    uv run python -m src.analysis.view_results --list

    # Show specific plot types
    uv run python -m src.analysis.view_results 20260101006 --plot prices
    uv run python -m src.analysis.view_results 20260101006 --plot convergence
    uv run python -m src.analysis.view_results 20260101006 --plot summary
    uv run python -m src.analysis.view_results 20260101006 --plot all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_results_dir() -> Path:
    """Get the results directory."""
    return Path("data/results")


def list_simulations() -> list[dict]:
    """List all available simulation results."""
    results_dir = get_results_dir()
    if not results_dir.exists():
        return []

    simulations = []
    for results_file in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            with open(results_file) as f:
                data = json.load(f)
            simulations.append({
                "id": results_file.stem,  # filename without .json
                "name": data.get("name"),
                "timestamp": data.get("timestamp", "unknown"),
                "config": data.get("config", {}),
                "total_trades": data.get("total_trades", 0),
                "avg_price": data.get("avg_price"),
                "equilibrium_price": data.get("equilibrium_price"),
            })
        except (json.JSONDecodeError, KeyError):
            pass
    return simulations


def load_results(experiment_id: str) -> dict:
    """Load results for a specific experiment."""
    results_file = get_results_dir() / f"{experiment_id}.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results found for experiment: {experiment_id}")

    with open(results_file) as f:
        return json.load(f)


def get_latest_experiment_id() -> str | None:
    """Get the ID of the most recent experiment."""
    simulations = list_simulations()
    if simulations:
        return simulations[0]["id"]
    return None


def print_simulation_list(simulations: list[dict]) -> None:
    """Print a formatted list of simulations."""
    if not simulations:
        print("No simulations found in data/results/")
        return

    print("\nAvailable Simulations:")
    print("=" * 95)
    print(f"{'ID':<15} {'Name':<20} {'Type':<8} {'Agents':<8} {'Rounds':<8} {'Trades':<8} {'Avg Price':<10}")
    print("-" * 95)

    for sim in simulations:
        config = sim["config"]
        name = sim.get("name") or "-"
        if len(name) > 18:
            name = name[:17] + "..."
        agent_type = config.get("agent_type", "?")
        n_agents = config.get("n_agents", "?")
        n_rounds = config.get("n_rounds", "?")
        trades = sim["total_trades"]
        avg_price = sim["avg_price"]
        avg_str = f"{avg_price:.2f}" if avg_price is not None else "N/A"

        print(f"{sim['id']:<15} {name:<20} {agent_type:<8} {n_agents:<8} {n_rounds:<8} {trades:<8} {avg_str:<10}")

    print("=" * 95)
    print(f"\nTotal: {len(simulations)} simulation(s)")
    print("\nTo view a simulation: uv run python -m src.analysis.view_results <ID>")


def print_summary(data: dict) -> None:
    """Print a text summary of the simulation results."""
    print("\n" + "=" * 60)
    print(f"Experiment: {data.get('experiment_id', 'unknown')}")
    if data.get("name"):
        print(f"Name: {data.get('name')}")
    print(f"Timestamp: {data.get('timestamp', 'unknown')}")
    print("=" * 60)

    config = data.get("config", {})
    print(f"\nConfiguration:")
    print(f"  Agent type: {config.get('agent_type', 'unknown')}")
    print(f"  Agents: {config.get('n_agents', 'unknown')}")
    print(f"  Rounds: {config.get('n_rounds', 'unknown')}")

    # Show agent setup if available
    agents = data.get("agents", [])
    if agents:
        print(f"\nAgent Setup (initial state):")
        for a in agents:
            role = a.get("role", "?").upper()
            val = a.get("valuation", "?")
            endow = a.get("initial_endowment", a.get("endowment", {}))
            money = endow.get("money", 0)
            goods = endow.get("good_A", 0)
            print(f"  {a.get('id')}: {role}, valuation={val}, money={money}, goods={goods}")

    print(f"\nResults:")
    print(f"  Total trades: {data.get('total_trades', 0)}")
    print(f"  Total volume: {data.get('total_volume', 0)}")
    eq_price = data.get('equilibrium_price')
    avg_price = data.get('avg_price')
    print(f"  Equilibrium price: {eq_price:.2f}" if eq_price else "  Equilibrium price: N/A")
    print(f"  Average price: {avg_price:.2f}" if avg_price else "  Average price: N/A")

    mad = data.get('convergence_mad')
    print(f"  Convergence (MAD): {mad:.2f}" if mad else "  Convergence (MAD): N/A")

    print("\nPer-Round Summary:")
    for rd in data.get("rounds", []):
        prices_str = ", ".join(f"{p:.1f}" for p in rd.get("prices", []))
        if not prices_str:
            prices_str = "no trades"
        print(f"  Round {rd['round']}: {rd['num_trades']} trade(s) - {prices_str}")


def show_plots(data: dict, plot_type: str) -> None:
    """Show the requested plot(s)."""
    from src.analysis.plotting import (
        plot_price_trajectory,
        plot_convergence,
        plot_market_summary,
    )

    round_data = data.get("rounds", [])
    eq_price = data.get("equilibrium_price")
    exp_id = data.get("experiment_id", "unknown")
    config = data.get("config", {})
    title_suffix = f" ({config.get('agent_type', '?')} agents, n={config.get('n_agents', '?')})"

    if plot_type in ("prices", "all"):
        plot_price_trajectory(
            round_data,
            equilibrium_price=eq_price,
            title=f"Price Trajectory - {exp_id}" + title_suffix
        )

    if plot_type in ("convergence", "all"):
        if eq_price is not None:
            plot_convergence(
                round_data,
                equilibrium_price=eq_price,
                title=f"Price Convergence - {exp_id}" + title_suffix
            )
        else:
            print("Cannot show convergence plot: no equilibrium price available")

    if plot_type in ("summary", "all"):
        plot_market_summary(
            round_data,
            equilibrium_price=eq_price
        )


def main():
    parser = argparse.ArgumentParser(
        description="View graphs from saved simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment ID to view (default: most recent)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available simulations"
    )
    parser.add_argument(
        "--plot", "-p",
        choices=["prices", "convergence", "summary", "all"],
        default="all",
        help="Which plot to show (default: all)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only show text summary, no plots"
    )
    parser.add_argument(
        "--round", "-r",
        type=int,
        default=None,
        help="Show detailed decisions for a specific round"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        simulations = list_simulations()
        print_simulation_list(simulations)
        return

    # Determine which experiment to view
    if args.experiment_id:
        exp_id = args.experiment_id
    else:
        exp_id = get_latest_experiment_id()
        if exp_id is None:
            print("No simulations found. Run a simulation first:")
            print("  uv run python main.py --agent-type react --agents 4 --rounds 5")
            return

    # Load and display results
    try:
        data = load_results(exp_id)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available simulations")
        return

    # Always show text summary
    print_summary(data)

    # Show round details if requested
    if args.round is not None:
        print_round_details(data, args.round)

    # Show plots unless --no-plot
    if not args.no_plot:
        show_plots(data, args.plot)


def print_round_details(data: dict, round_num: int) -> None:
    """Print detailed decisions for a specific round."""
    rounds = data.get("rounds", [])
    round_data = None
    for rd in rounds:
        if rd["round"] == round_num:
            round_data = rd
            break

    if round_data is None:
        print(f"\nRound {round_num} not found.")
        return

    print(f"\n{'=' * 60}")
    print(f"Round {round_num} Details")
    print("=" * 60)

    prices_str = ", ".join(f"{p:.2f}" for p in round_data.get("prices", []))
    print(f"Trades: {round_data['num_trades']} | Prices: {prices_str or 'none'}")

    decisions = round_data.get("decisions", [])
    if not decisions:
        print("\nNo decision logs available for this round.")
        return

    print(f"\nAgent Decisions:")
    print("-" * 60)

    for dec in decisions:
        agent_id = dec.get("agent_id", "?")
        order = dec.get("order")
        decision = dec.get("decision", {})

        if order:
            action_str = f"{order['side'].upper()} @ {order['price']:.2f}"
        else:
            action_str = f"{decision.get('action', 'hold').upper()} (no order)"

        print(f"\n{agent_id}: {action_str}")

        # Show reasoning if available (CoT mode)
        analysis = dec.get("analysis", "")
        reasoning = dec.get("reasoning", "")

        if analysis:
            print(f"  Analysis: {analysis[:100]}..." if len(analysis) > 100 else f"  Analysis: {analysis}")
        if reasoning:
            print(f"  Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"  Reasoning: {reasoning}")


if __name__ == "__main__":
    main()
