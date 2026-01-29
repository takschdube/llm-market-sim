# src/simulation/parallel.py
"""
Parallel trial execution for experiments.

This module provides utilities to run multiple independent trials concurrently,
significantly speeding up experiment execution.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Any, Optional


def run_trials_parallel(
    trial_fn: Callable[[int], Dict],
    n_trials: int,
    max_workers: Optional[int] = None,
    start_trial: int = 0,
    progress_callback: Optional[Callable[[int, Dict], None]] = None,
) -> List[Dict]:
    """
    Run multiple trials in parallel using ThreadPoolExecutor.

    Args:
        trial_fn: Function that takes trial_id and returns result dict
        n_trials: Total number of trials to run
        max_workers: Maximum concurrent trials (default: min(n_trials, 4))
        start_trial: Starting trial ID (for resume support)
        progress_callback: Called after each trial completes with (trial_id, result)

    Returns:
        List of result dicts, ordered by trial_id

    Example:
        def run_trial(trial_id):
            # ... run simulation ...
            return {"trial_id": trial_id, "mad": 1.23}

        results = run_trials_parallel(run_trial, n_trials=30, max_workers=4)
    """
    if max_workers is None:
        # Default to 4 concurrent trials, or fewer if n_trials is small
        max_workers = min(n_trials - start_trial, 4)

    # Ensure we don't exceed reasonable limits
    max_workers = min(max_workers, 8)  # Cap at 8 to avoid API rate limits

    trials_to_run = list(range(start_trial, n_trials))
    results = [None] * len(trials_to_run)

    print(f"  Running {len(trials_to_run)} trials with {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all trials
        future_to_trial = {
            executor.submit(trial_fn, trial_id): trial_id
            for trial_id in trials_to_run
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_trial):
            trial_id = future_to_trial[future]
            try:
                result = future.result()
                results[trial_id - start_trial] = result
                completed += 1

                if progress_callback:
                    progress_callback(trial_id, result)
                else:
                    # Default progress indicator
                    print(f"    Completed trial {trial_id + 1}/{n_trials} "
                          f"({completed}/{len(trials_to_run)} in batch)")

            except Exception as e:
                print(f"    Trial {trial_id} failed: {e}")
                results[trial_id - start_trial] = {
                    "trial_id": trial_id,
                    "error": str(e),
                }

    return results


def run_conditions_parallel(
    condition_fn: Callable[[str], List[Dict]],
    conditions: List[str],
    max_workers: int = 2,
) -> Dict[str, List[Dict]]:
    """
    Run multiple conditions in parallel.

    This is useful when conditions are independent (e.g., ZI vs React vs CoT).
    Be careful with API rate limits when using this.

    Args:
        condition_fn: Function that takes condition name and returns list of trial results
        conditions: List of condition names to run
        max_workers: Maximum concurrent conditions (default: 2)

    Returns:
        Dict mapping condition names to lists of trial results
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_condition = {
            executor.submit(condition_fn, cond): cond
            for cond in conditions
        }

        for future in as_completed(future_to_condition):
            condition = future_to_condition[future]
            try:
                results[condition] = future.result()
                print(f"  Completed condition: {condition}")
            except Exception as e:
                print(f"  Condition {condition} failed: {e}")
                results[condition] = []

    return results
