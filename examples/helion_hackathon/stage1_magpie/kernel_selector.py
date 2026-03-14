"""
Kernel Selector for Stage 1: Ranking kernels by autotuning potential.

Implements the selection criterion from the hackathon plan:
selection_score = wall_time_fraction * log(config_space_size)

This identifies kernels where adaptive autotuning has maximum leverage.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

from .kernel_profiler import ProfileResult


@dataclass
class SelectionResult:
    """Result of kernel selection with ranking scores."""
    kernel_name: str
    wall_time_fraction: float
    config_space_size: int
    selection_score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "wall_time_fraction": self.wall_time_fraction,
            "config_space_size": self.config_space_size,
            "selection_score": self.selection_score,
            "rank": self.rank,
        }


class KernelSelector:
    """
    Selects top kernels for autotuning based on wall-time and config space.

    The selection score prioritizes kernels that:
    1. Account for significant execution time (high wall-time fraction)
    2. Have large configuration spaces (more room for optimization)

    Formula: score = wall_time_fraction * log(config_space_size)
    """

    def __init__(
        self,
        min_wall_time_fraction: float = 0.01,
        min_config_space_size: int = 10,
    ):
        """
        Initialize the selector.

        Args:
            min_wall_time_fraction: Minimum wall-time % to consider (default 1%)
            min_config_space_size: Minimum config space to consider
        """
        self.min_wall_time_fraction = min_wall_time_fraction
        self.min_config_space_size = min_config_space_size

    def compute_selection_score(
        self,
        wall_time_fraction: float,
        config_space_size: int,
    ) -> float:
        """
        Compute the selection score for a kernel.

        Args:
            wall_time_fraction: Fraction of total wall time (0 to 1)
            config_space_size: Number of possible configurations

        Returns:
            Selection score (higher = better candidate for autotuning)
        """
        if config_space_size <= 1:
            return 0.0

        # log(config_space) captures the "room for optimization"
        # Multiplying by wall_time_fraction weights by actual impact
        log_space = math.log(config_space_size)
        return wall_time_fraction * log_space

    def select_top_kernels(
        self,
        profile_results: List[ProfileResult],
        top_k: int = 5,
    ) -> List[SelectionResult]:
        """
        Select top-k kernels for autotuning.

        Args:
            profile_results: List of kernel profiling results
            top_k: Number of kernels to select

        Returns:
            List of SelectionResult, sorted by selection score
        """
        candidates = []

        for profile in profile_results:
            # Filter by minimum thresholds
            if profile.wall_time_fraction < self.min_wall_time_fraction:
                continue
            if profile.config_space_size < self.min_config_space_size:
                continue

            score = self.compute_selection_score(
                profile.wall_time_fraction,
                profile.config_space_size,
            )

            candidates.append((profile, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Create selection results
        results = []
        for rank, (profile, score) in enumerate(candidates[:top_k], start=1):
            results.append(SelectionResult(
                kernel_name=profile.kernel_name,
                wall_time_fraction=profile.wall_time_fraction,
                config_space_size=profile.config_space_size,
                selection_score=score,
                rank=rank,
            ))

        return results

    def explain_selection(self, results: List[SelectionResult]) -> str:
        """
        Generate a human-readable explanation of the selection.

        Returns a formatted string suitable for demo slides.
        """
        lines = []
        lines.append("=" * 90)
        lines.append("KERNEL SELECTION RESULTS - Ranked by Autotuning Potential")
        lines.append("Formula: selection_score = wall_time_fraction * log(config_space_size)")
        lines.append("=" * 90)
        lines.append("")
        lines.append(f"{'Rank':<6} {'Kernel Name':<25} {'Wall-time %':>12} {'Config Space':>15} {'Score':>12}")
        lines.append("-" * 90)

        total_wall_time = sum(r.wall_time_fraction for r in results)

        for r in results:
            space_str = f"{r.config_space_size:.2e}" if r.config_space_size > 1000 else str(r.config_space_size)
            lines.append(
                f"#{r.rank:<5} {r.kernel_name:<25} {r.wall_time_fraction*100:>11.2f}% "
                f"{space_str:>15} {r.selection_score:>12.4f}"
            )

        lines.append("-" * 90)
        lines.append(f"Total wall-time coverage: {total_wall_time*100:.1f}%")
        lines.append("")
        lines.append("Selection rationale:")
        lines.append(f"  - These {len(results)} kernels represent {total_wall_time*100:.1f}% of model execution time")
        lines.append("  - Large config spaces provide room for adaptive autotuning to find better configs")
        lines.append("  - High wall-time kernels give maximum latency improvement per optimization effort")
        lines.append("=" * 90)

        return "\n".join(lines)

    @staticmethod
    def format_demo_table(results: List[SelectionResult]) -> str:
        """
        Format results as a simple table for demo slides.

        Returns:
            Markdown-formatted table string
        """
        lines = []
        lines.append("| Kernel | Wall-time % | Config Space | Selection Score |")
        lines.append("|--------|-------------|--------------|-----------------|")

        for r in results:
            space_str = f"{r.config_space_size:.1e}"
            lines.append(
                f"| {r.kernel_name} | {r.wall_time_fraction*100:.1f}% | {space_str} | {r.selection_score:.3f} |"
            )

        return "\n".join(lines)


def rank_kernels(
    profile_results: List[ProfileResult],
    top_k: int = 5,
) -> Tuple[List[SelectionResult], str]:
    """
    Convenience function to rank kernels and get explanation.

    Args:
        profile_results: Kernel profiling results
        top_k: Number of kernels to select

    Returns:
        Tuple of (selection_results, explanation_string)
    """
    selector = KernelSelector()
    results = selector.select_top_kernels(profile_results, top_k=top_k)
    explanation = selector.explain_selection(results)
    return results, explanation
