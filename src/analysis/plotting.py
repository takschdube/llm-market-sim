# src/analysis/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_price_trajectory(
    round_data: List[dict],
    equilibrium_price: Optional[float] = None,
    title: str = "Price Trajectory",
    save_path: Optional[str] = None
) -> None:
    """
    Plot transaction prices over rounds.

    Args:
        round_data: List of dicts with 'round' and 'prices' keys
        equilibrium_price: Optional Walrasian equilibrium price to show as reference
        title: Plot title
        save_path: If provided, save figure to this path
    """
    _, ax = plt.subplots(figsize=(10, 6))

    # Collect all prices with their round numbers
    rounds = []
    prices = []
    for rd in round_data:
        for price in rd["prices"]:
            rounds.append(rd["round"])
            prices.append(price)

    if prices:
        ax.scatter(rounds, prices, alpha=0.6, s=50, label="Transaction prices")

        # Add moving average trend line
        if len(prices) >= 3:
            # Compute average price per round
            round_avg = {}
            for r, p in zip(rounds, prices):
                round_avg.setdefault(r, []).append(p)
            avg_rounds = sorted(round_avg.keys())
            avg_prices = [np.mean(round_avg[r]) for r in avg_rounds]
            ax.plot(avg_rounds, avg_prices, 'b-', linewidth=2, label="Round average")

    # Show equilibrium price
    if equilibrium_price is not None:
        ax.axhline(y=equilibrium_price, color='r', linestyle='--',
                   linewidth=2, label=f"Walrasian eq. (p*={equilibrium_price:.2f})")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_efficiency_over_time(
    round_data: List[dict],
    title: str = "Allocative Efficiency Over Time",
    save_path: Optional[str] = None
) -> None:
    """
    Plot cumulative allocative efficiency over rounds.
    Requires round_data to have 'efficiency' key (added by simulation runner).
    """
    _, ax = plt.subplots(figsize=(10, 6))

    efficiencies = []
    rounds = []

    for rd in round_data:
        rounds.append(rd["round"])
        efficiencies.append(rd.get("efficiency", 0))

    if efficiencies and any(e > 0 for e in efficiencies):
        ax.plot(rounds, efficiencies, 'g-o', linewidth=2, markersize=6)
        ax.axhline(y=1.0, color='r', linestyle='--', label="Perfect efficiency")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Efficiency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_convergence(
    round_data: List[dict],
    equilibrium_price: float,
    title: str = "Price Convergence to Equilibrium",
    save_path: Optional[str] = None
) -> None:
    """
    Plot mean absolute deviation from equilibrium over rounds.
    """
    _, ax = plt.subplots(figsize=(10, 6))

    rounds = []
    deviations = []

    for rd in round_data:
        if rd["prices"]:
            rounds.append(rd["round"])
            mad = np.mean([abs(p - equilibrium_price) for p in rd["prices"]])
            deviations.append(mad)

    if deviations:
        ax.plot(rounds, deviations, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Mean Absolute Deviation from p*", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_market_summary(
    round_data: List[dict],
    equilibrium_price: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a summary figure with price trajectory and volume.
    """
    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Price trajectory
    ax1 = axes[0]
    rounds = []
    prices = []
    for rd in round_data:
        for price in rd["prices"]:
            rounds.append(rd["round"])
            prices.append(price)

    if prices:
        ax1.scatter(rounds, prices, alpha=0.6, s=50)

    if equilibrium_price is not None:
        ax1.axhline(y=equilibrium_price, color='r', linestyle='--',
                    linewidth=2, label=f"p*={equilibrium_price:.2f}")
        ax1.legend()

    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title("Market Activity Summary", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Bottom: Volume per round
    ax2 = axes[1]
    round_nums = [rd["round"] for rd in round_data]
    volumes = [rd["volume"] for rd in round_data]

    ax2.bar(round_nums, volumes, alpha=0.7, color='steelblue')
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()
