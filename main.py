"""
LLM Market Simulation
=====================
Simulates trading markets with LLM-powered agents to study
emergent price dynamics and market efficiency.

Usage:
    python main.py                        # Default: 4 react agents, 10 rounds
    python main.py --agents 6 --rounds 20
    python main.py --agent-type zi        # Zero-intelligence baseline
    python main.py --agent-type cot       # Chain-of-thought reasoning agents

Agent Types:
    zi    - Zero-Intelligence (Gode & Sunder 1993 random baseline)
    react - LLM with reactive reasoning (observe → act)
    cot   - LLM with Chain-of-Thought (observe → analyze → reason → act)

All LLM agents use LangGraph with configurable reasoning depth.
Results saved to: data/results/<experiment_id>.json
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.simulation.runner import Simulation, create_agents_from_scheme
from src.simulation.valuations import get_scheme, VALUATION_SCHEMES
from src.agents import ReactAgent, CoTAgent, list_agents
from src.analysis.metrics import compute_walrasian_price, compute_convergence
from src.analysis.plotting import plot_price_trajectory


def generate_experiment_id(results_dir: Path) -> str:
    """Generate experiment ID in format YYYYMMDDNNN (e.g., 20260101001)."""
    today = datetime.now().strftime("%Y%m%d")

    # Find existing experiments from today (now .json files, not directories)
    existing = list(results_dir.glob(f"{today}*.json"))

    # Get the next number
    if existing:
        numbers = []
        for p in existing:
            try:
                # Get NNN part from filename like "20260101001.json"
                num = int(p.stem[8:])
                numbers.append(num)
            except (ValueError, IndexError):
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    return f"{today}{next_num:03d}"


def main():
    parser = argparse.ArgumentParser(description="LLM Market Simulation")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of agents (half buyers, half sellers)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of trading rounds")
    parser.add_argument("--agent-type", choices=["zi", "react", "cot"], default="react",
                        help="Agent type: 'zi' (zero-intelligence), 'react' (LLM reactive), 'cot' (LLM chain-of-thought)")
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "anthropic", "openai", "google"],
                        help="LLM provider for react/cot agents")
    parser.add_argument("--model", default=None,
                        help="Model name (defaults to provider's default)")
    parser.add_argument("--output-dir", type=str, default="data/results",
                        help="Base directory for experiment results")
    parser.add_argument("--name", type=str, default=None,
                        help="Optional name/description for this experiment")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--valuation-scheme",
                        choices=list(VALUATION_SCHEMES.keys()),
                        default="linear",
                        help="Valuation distribution scheme (default: linear)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (used by uniform scheme)")
    args = parser.parse_args()

    print("=" * 50)
    print("LLM Market Simulation")
    print("=" * 50)
    print(f"Agents: {args.agents} ({args.agent_type})")
    print(f"Rounds: {args.rounds}")
    print(f"Valuation scheme: {args.valuation_scheme}")
    if args.agent_type != "zi":
        print(f"Provider: {args.provider}")
        if args.model:
            print(f"Model: {args.model}")
    print()

    # Create valuation scheme
    scheme_kwargs = {}
    if args.seed is not None:
        scheme_kwargs["seed"] = args.seed
    valuation_scheme = get_scheme(args.valuation_scheme, **scheme_kwargs)

    # Create agents using the new registry-based function
    agents = create_agents_from_scheme(
        n_agents=args.agents,
        agent_type=args.agent_type,
        provider=args.provider,
        model=args.model,
        valuation_scheme=valuation_scheme
    )

    # Compute theoretical equilibrium price
    eq_price = compute_walrasian_price(agents)
    print(f"Theoretical Walrasian equilibrium price: {eq_price:.2f}")
    print()

    # Capture initial agent states BEFORE simulation modifies them
    agent_setup = []
    print("Agent Setup:")
    for agent in agents:
        has_money = agent.state.endowment.get("money", 0) > 0
        role = "Buyer" if has_money else "Seller"
        val = agent.state.valuation.get("good_A", 0)
        print(f"  {agent.state.id}: {role}, valuation={val}")
        agent_setup.append({
            "id": agent.state.id,
            "role": role.lower(),
            "valuation": val,
            "initial_endowment": agent.state.endowment.copy(),
        })
    print()

    # Run simulation
    print("Running simulation...")
    sim = Simulation(agents, num_rounds=args.rounds)
    results = sim.run()

    # Compute metrics
    all_prices = [p for r in results for p in r["prices"]]
    total_trades = sum(r["num_trades"] for r in results)
    total_volume = sum(r["volume"] for r in results)

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total trades: {total_trades}")
    print(f"Total volume: {total_volume:.1f}")

    convergence = None
    avg_price = None
    if all_prices:
        import numpy as np
        avg_price = np.mean(all_prices)
        std_price = np.std(all_prices)
        convergence = compute_convergence(all_prices, eq_price)

        print(f"Average price: {avg_price:.2f}")
        print(f"Price std dev: {std_price:.2f}")
        print(f"Equilibrium price: {eq_price:.2f}")
        print(f"Convergence (MAD): {convergence:.2f}")

        # Plot if requested
        if not args.no_plot:
            plot_price_trajectory(
                results,
                equilibrium_price=eq_price,
                title=f"Market Simulation ({args.agent_type.upper()} Agents)"
            )
    else:
        print("No trades occurred!")

    # Generate experiment ID and create output directory
    results_base = Path(args.output_dir)
    results_base.mkdir(parents=True, exist_ok=True)
    experiment_id = generate_experiment_id(results_base)

    # Collect decision logs from agents, organized by round
    decision_logs_by_round = {}
    for agent in agents:
        for log in agent.get_logs():
            round_num = log.get("round", 0)
            if round_num not in decision_logs_by_round:
                decision_logs_by_round[round_num] = []
            decision_logs_by_round[round_num].append(log)

    # Merge decision logs into round data (remove trade objects for JSON serialization)
    for rd in results:
        round_num = rd["round"]
        rd["decisions"] = decision_logs_by_round.get(round_num, [])
        # Remove trade objects (not JSON serializable)
        if "trades" in rd:
            del rd["trades"]

    output_data = {
        "experiment_id": experiment_id,
        "name": args.name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_agents": args.agents,
            "n_rounds": args.rounds,
            "agent_type": args.agent_type,
            "provider": args.provider if args.agent_type != "zi" else None,
            "model": args.model,
        },
        "valuation_scheme": valuation_scheme.to_dict(),
        "agents": agent_setup,
        "equilibrium_price": eq_price,
        "total_trades": total_trades,
        "total_volume": total_volume,
        "convergence_mad": convergence,
        "avg_price": float(avg_price) if avg_price is not None else None,
        "rounds": results,
    }

    # Save results as single JSON file
    results_file = results_base / f"{experiment_id}.json"
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
