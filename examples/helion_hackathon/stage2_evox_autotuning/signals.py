"""
Search State Signals for Adaptive Autotuning.

Implements the three observable signals from the hackathon plan:
1. improvement_velocity: Rolling average of latency delta over last K evaluations
2. population_diversity: Mean Hamming distance between recently sampled configs
3. error_pressure: Compile error/timeout rate in last batch

These signals drive the adaptive state machine transitions.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Deque, Optional
from collections import deque
from enum import Enum
import math


class SearchMode(Enum):
    """
    Search modes for the adaptive controller.

    EXPLOIT: Focus on local refinement around best configs
    EXPLORE: Broaden search to find new promising regions
    ESCAPE: Force random restart from unexplored region
    """
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    ESCAPE = "escape"


@dataclass
class SearchState:
    """Current state of the search process."""
    mode: SearchMode = SearchMode.EXPLORE
    improvement_velocity: float = 0.0
    population_diversity: float = 0.0
    error_pressure: float = 0.0
    generation: int = 0
    total_evaluations: int = 0
    best_latency: float = float("inf")


@dataclass
class SignalConfig:
    """Configuration for signal computation."""
    # Window size for rolling averages
    velocity_window: int = 20
    diversity_window: int = 50
    error_window: int = 30

    # Thresholds for mode transitions
    high_velocity_threshold: float = 0.01  # 1% improvement per eval
    low_velocity_threshold: float = 0.001  # 0.1% improvement per eval
    high_diversity_threshold: float = 0.5  # 50% of max possible
    low_diversity_threshold: float = 0.1   # 10% of max possible
    high_error_threshold: float = 0.3      # 30% error rate


class SignalTracker:
    """
    Tracks and computes search state signals.

    This class observes the autotuning process and computes the three
    key signals that drive adaptive strategy switching:

    1. improvement_velocity: How fast are we finding better configs?
       - High velocity → stay in EXPLOIT mode
       - Low velocity → consider EXPLORE or ESCAPE

    2. population_diversity: How spread out are our recent samples?
       - High diversity + low velocity → EXPLORE (still finding new regions)
       - Low diversity + low velocity → ESCAPE (stuck in local optimum)

    3. error_pressure: How often are configs failing to compile/run?
       - High error rate suggests we're in a difficult region
       - May need to ESCAPE to a more stable region
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialize the signal tracker.

        Args:
            config: Configuration for signal computation
        """
        self.config = config or SignalConfig()

        # Rolling windows for each signal
        self._latency_history: Deque[float] = deque(maxlen=self.config.velocity_window)
        self._diversity_history: Deque[float] = deque(maxlen=self.config.diversity_window)
        self._error_history: Deque[bool] = deque(maxlen=self.config.error_window)

        # Config encoding history for diversity computation
        self._config_history: Deque[Tuple[int, ...]] = deque(maxlen=self.config.diversity_window)

        # Best latency tracking
        self._best_latency = float("inf")
        self._total_evaluations = 0
        self._generation = 0

    def record_evaluation(
        self,
        latency: float,
        config_encoding: Tuple[int, ...],
        is_error: bool = False,
    ):
        """
        Record the result of evaluating a single configuration.

        Args:
            latency: Measured latency (inf if error/timeout)
            config_encoding: Encoded configuration tuple
            is_error: Whether the evaluation resulted in error/timeout
        """
        self._total_evaluations += 1
        self._error_history.append(is_error)
        self._config_history.append(config_encoding)

        if not is_error and latency < float("inf"):
            self._latency_history.append(latency)
            if latency < self._best_latency:
                self._best_latency = latency

    def record_batch(
        self,
        results: List[Tuple[float, Tuple[int, ...], bool]],
    ):
        """
        Record results from a batch of evaluations.

        Args:
            results: List of (latency, config_encoding, is_error) tuples
        """
        for latency, encoding, is_error in results:
            self.record_evaluation(latency, encoding, is_error)

    def set_generation(self, generation: int):
        """Set the current generation number."""
        self._generation = generation

    def compute_improvement_velocity(self) -> float:
        """
        Compute the improvement velocity signal.

        This measures the rate of latency improvement over recent evaluations.
        Computed as: (old_best - new_best) / old_best / num_evals

        Returns:
            Improvement velocity (0 to ~1, higher = faster improvement)
        """
        if len(self._latency_history) < 2:
            return 0.0

        history = list(self._latency_history)

        # Compare first half to second half
        mid = len(history) // 2
        if mid == 0:
            return 0.0

        old_best = min(history[:mid])
        new_best = min(history[mid:])

        if old_best <= 0 or old_best == float("inf"):
            return 0.0

        # Relative improvement per evaluation
        improvement = (old_best - new_best) / old_best
        velocity = improvement / (len(history) - mid)

        return max(0.0, velocity)

    def compute_population_diversity(self) -> float:
        """
        Compute the population diversity signal.

        This measures how spread out our recent samples are in config space.
        Uses mean Hamming distance between all pairs of recent configs.

        Returns:
            Diversity score (0 to 1, higher = more diverse)
        """
        if len(self._config_history) < 2:
            return 1.0  # Assume high diversity at start

        configs = list(self._config_history)

        # Compute mean pairwise Hamming distance
        total_distance = 0.0
        num_pairs = 0
        max_possible_distance = len(configs[0]) if configs else 1

        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                distance = self._hamming_distance(configs[i], configs[j])
                total_distance += distance
                num_pairs += 1

        if num_pairs == 0 or max_possible_distance == 0:
            return 1.0

        # Normalize by max possible distance
        mean_distance = total_distance / num_pairs
        normalized = mean_distance / max_possible_distance

        return min(1.0, normalized)

    def compute_error_pressure(self) -> float:
        """
        Compute the error pressure signal.

        This measures the rate of compile errors and timeouts.

        Returns:
            Error rate (0 to 1, higher = more errors)
        """
        if not self._error_history:
            return 0.0

        return sum(self._error_history) / len(self._error_history)

    def get_current_state(self) -> SearchState:
        """
        Get the current search state with all computed signals.

        Returns:
            SearchState with current signal values and mode
        """
        velocity = self.compute_improvement_velocity()
        diversity = self.compute_population_diversity()
        error = self.compute_error_pressure()

        # Determine mode based on signals
        mode = self._determine_mode(velocity, diversity, error)

        return SearchState(
            mode=mode,
            improvement_velocity=velocity,
            population_diversity=diversity,
            error_pressure=error,
            generation=self._generation,
            total_evaluations=self._total_evaluations,
            best_latency=self._best_latency,
        )

    def _determine_mode(
        self,
        velocity: float,
        diversity: float,
        error: float,
    ) -> SearchMode:
        """
        Determine the search mode based on signals.

        State machine logic from hackathon plan:
        - High velocity, any diversity → EXPLOIT
        - Low velocity, high diversity → EXPLORE
        - Low velocity, low diversity → ESCAPE
        - High error pressure → ESCAPE (override)
        """
        cfg = self.config

        # High error pressure forces ESCAPE
        if error > cfg.high_error_threshold:
            return SearchMode.ESCAPE

        # High velocity → EXPLOIT (things are working)
        if velocity > cfg.high_velocity_threshold:
            return SearchMode.EXPLOIT

        # Low velocity conditions
        if velocity < cfg.low_velocity_threshold:
            # Check diversity to decide between EXPLORE and ESCAPE
            if diversity > cfg.high_diversity_threshold:
                return SearchMode.EXPLORE  # Still finding new regions
            elif diversity < cfg.low_diversity_threshold:
                return SearchMode.ESCAPE   # Stuck in local optimum
            else:
                return SearchMode.EXPLORE  # Default to explore

        # Medium velocity → EXPLOIT (slight preference)
        return SearchMode.EXPLOIT

    @staticmethod
    def _hamming_distance(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        """Compute Hamming distance between two encoded configs."""
        if len(a) != len(b):
            return max(len(a), len(b))
        return sum(x != y for x, y in zip(a, b))

    def format_status(self) -> str:
        """Format current status as a human-readable string."""
        state = self.get_current_state()
        return (
            f"[Gen {state.generation:3d}] "
            f"Mode: {state.mode.value:7s} | "
            f"Velocity: {state.improvement_velocity:.4f} | "
            f"Diversity: {state.population_diversity:.3f} | "
            f"Errors: {state.error_pressure:.2%} | "
            f"Best: {state.best_latency:.4f}ms"
        )
