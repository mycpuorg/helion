#!/usr/bin/env python
"""
Visualize OpenEvolve autotuning candidates.

Usage:
    uv run python examples/visualize_openevolve_candidates.py [artifacts_dir]

This creates several visualizations:
1. Score progression over iterations
2. Best score over time (cumulative best)
3. Parameter exploration heatmap
4. Config comparison table
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_tuning_data(artifacts_dir: Path) -> dict:
    """Load tuning summary from artifacts directory."""
    summary_path = artifacts_dir / "tuning_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No tuning_summary.json found in {artifacts_dir}")

    with open(summary_path) as f:
        return json.load(f)


def load_eval_files(artifacts_dir: Path) -> list[dict]:
    """Load individual evaluation files."""
    evals = []
    for eval_file in sorted(artifacts_dir.glob("eval_*.json")):
        with open(eval_file) as f:
            evals.append(json.load(f))
    return evals


def plot_score_progression(history: list[dict], ax: plt.Axes) -> None:
    """Plot score progression over iterations."""
    iterations = list(range(len(history)))
    scores = [h["score"] for h in history]
    best_so_far = [h["best_so_far"] for h in history]

    ax.scatter(iterations, scores, c="blue", alpha=0.6, s=50, label="Score", zorder=3)
    ax.plot(iterations, best_so_far, c="red", linewidth=2, label="Best so far", zorder=2)
    ax.axhline(y=max(scores), color="green", linestyle="--", alpha=0.5, label=f"Best: {max(scores):.2f}")

    # Mark failed configs (score=0)
    failed = [(i, s) for i, s in enumerate(scores) if s == 0]
    if failed:
        fail_x, fail_y = zip(*failed)
        ax.scatter(fail_x, [0.5] * len(fail_x), c="red", marker="x", s=100, label="Failed", zorder=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (1/latency_ms)")
    ax.set_title("OpenEvolve Score Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_parameter_exploration(history: list[dict], ax: plt.Axes) -> None:
    """Plot which parameter values were explored."""
    # Extract parameter values
    params = {
        "block_m": [],
        "block_n": [],
        "block_k": [],
        "num_warps": [],
        "num_stages": [],
    }
    scores = []

    for h in history:
        config = h["config"]
        for param in params:
            params[param].append(config.get(param, 0))
        scores.append(h["score"])

    # Create scatter matrix-like visualization
    param_names = list(params.keys())
    n_params = len(param_names)

    # Normalize scores for coloring
    scores_arr = np.array(scores)
    scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-10)

    # Plot block_m vs block_n colored by score
    scatter = ax.scatter(
        params["block_m"],
        params["block_n"],
        c=scores,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black"
    )

    # Add iteration numbers
    for i, (x, y) in enumerate(zip(params["block_m"], params["block_n"])):
        ax.annotate(str(i), (x, y), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("block_m")
    ax.set_ylabel("block_n")
    ax.set_title("Block Size Exploration (colored by score)")
    plt.colorbar(scatter, ax=ax, label="Score")
    ax.grid(True, alpha=0.3)


def plot_parameter_distribution(history: list[dict], ax: plt.Axes) -> None:
    """Plot parameter value distribution as stacked bars."""
    params_to_plot = ["block_m", "block_n", "block_k", "num_warps", "num_stages"]

    # Count occurrences of each value
    param_counts = {}
    for param in params_to_plot:
        values = [h["config"].get(param, 0) for h in history]
        unique_vals = sorted(set(values))
        counts = {v: values.count(v) for v in unique_vals}
        param_counts[param] = counts

    # Create bar chart
    x = np.arange(len(params_to_plot))
    width = 0.8

    for i, param in enumerate(params_to_plot):
        counts = param_counts[param]
        bottom = 0
        for val, count in sorted(counts.items()):
            ax.bar(i, count, width, bottom=bottom, label=f"{val}" if i == 0 else "")
            ax.text(i, bottom + count/2, f"{val}", ha="center", va="center", fontsize=8)
            bottom += count

    ax.set_xticks(x)
    ax.set_xticklabels(params_to_plot, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Parameter Value Distribution")


def plot_latency_comparison(history: list[dict], ax: plt.Axes) -> None:
    """Plot latency (1/score) comparison."""
    iterations = list(range(len(history)))
    latencies = [1000/h["score"] if h["score"] > 0 else None for h in history]

    # Filter out None values for plotting
    valid_points = [(i, l) for i, l in zip(iterations, latencies) if l is not None]
    if valid_points:
        valid_x, valid_y = zip(*valid_points)
        ax.bar(valid_x, valid_y, color="steelblue", alpha=0.7)

        # Highlight best
        best_idx = min(range(len(valid_y)), key=lambda i: valid_y[i])
        ax.bar(valid_x[best_idx], valid_y[best_idx], color="green", alpha=0.9, label=f"Best: {valid_y[best_idx]:.3f}ms")

    # Mark failed
    failed_x = [i for i, l in zip(iterations, latencies) if l is None]
    if failed_x:
        ax.bar(failed_x, [0.01] * len(failed_x), color="red", alpha=0.7, label="Failed")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Kernel Latency per Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def print_config_table(history: list[dict]) -> None:
    """Print a table of all configurations."""
    print("\n" + "="*120)
    print("Configuration History")
    print("="*120)
    print(f"{'Iter':>4} | {'Score':>8} | {'Latency':>10} | {'block_m':>7} | {'block_n':>7} | {'block_k':>7} | {'warps':>5} | {'stages':>6} | {'pid_type':>20}")
    print("-"*120)

    for i, h in enumerate(history):
        config = h["config"]
        score = h["score"]
        latency = f"{1000/score:.3f}ms" if score > 0 else "FAILED"

        print(f"{i:>4} | {score:>8.2f} | {latency:>10} | {config.get('block_m', 0):>7} | {config.get('block_n', 0):>7} | {config.get('block_k', 0):>7} | {config.get('num_warps', 0):>5} | {config.get('num_stages', 0):>6} | {config.get('pid_type', 'N/A'):>20}")

    print("="*120)

    # Summary
    valid_scores = [h["score"] for h in history if h["score"] > 0]
    if valid_scores:
        best_score = max(valid_scores)
        best_idx = next(i for i, h in enumerate(history) if h["score"] == best_score)
        print(f"\nBest configuration: Iteration {best_idx}")
        print(f"Best score: {best_score:.4f} (latency: {1000/best_score:.4f}ms)")
        print(f"Total evaluations: {len(history)}")
        print(f"Failed evaluations: {len(history) - len(valid_scores)}")


def main():
    # Default artifacts directory
    default_dir = Path("/home/manoj/software/helion/openevolve_benchmark_artifacts")
    artifacts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    print(f"Loading data from: {artifacts_dir}")

    # Load data
    data = load_tuning_data(artifacts_dir)
    history = data["history"]

    print(f"Found {len(history)} evaluations")
    print(f"Best score: {data['best_score']:.4f}")

    # Print text table
    print_config_table(history)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OpenEvolve Autotuning Visualization", fontsize=14, fontweight="bold")

    # Plot 1: Score progression
    plot_score_progression(history, axes[0, 0])

    # Plot 2: Parameter exploration (block sizes)
    plot_parameter_exploration(history, axes[0, 1])

    # Plot 3: Parameter distribution
    plot_parameter_distribution(history, axes[1, 0])

    # Plot 4: Latency comparison
    plot_latency_comparison(history, axes[1, 1])

    plt.tight_layout()

    # Save figure
    output_path = artifacts_dir / "visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
