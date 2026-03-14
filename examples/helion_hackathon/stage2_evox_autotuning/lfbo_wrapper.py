"""
Adaptive LFBO Pattern Search Wrapper.

Extends Helion's LFBOPatternSearch with adaptive parameter control
based on EvoX-style state machine signals.

This is the primary integration point with Helion's autotuner.
"""
import sys
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable

# Add Helion to path if available
HELION_PATH = os.path.expanduser("~/software/helion")
if os.path.exists(HELION_PATH):
    sys.path.insert(0, HELION_PATH)

from .signals import SignalTracker, SearchState, SearchMode
from .adaptive_controller import AdaptiveController, ControllerConfig, AdaptiveParameters


@dataclass
class AutotuneResult:
    """Result from an autotuning run."""
    best_config: Dict[str, Any]
    best_latency: float
    total_evaluations: int
    total_time_seconds: float
    latency_history: List[float]
    time_history: List[float]
    config_history: List[Dict[str, Any]]
    controller_history: Dict[str, List[Any]]


class AdaptiveLFBOPatternSearch:
    """
    Adaptive LFBO Pattern Search with EvoX-style strategy switching.

    This class wraps Helion's autotuning infrastructure and adds:
    1. Signal tracking (improvement velocity, diversity, error pressure)
    2. Adaptive parameter control (perturbation radius, classifier threshold)
    3. Mode switching (EXPLOIT, EXPLORE, ESCAPE)

    The key insight is that fixed search strategies are suboptimal -
    the search should adapt based on its current state.
    """

    def __init__(
        self,
        kernel_fn: Callable,
        config_space: List[Dict[str, Any]],
        benchmark_fn: Callable[[Dict[str, Any]], Tuple[float, bool]],
        encode_fn: Callable[[Dict[str, Any]], Tuple[int, ...]],
        controller_config: Optional[ControllerConfig] = None,
    ):
        """
        Initialize the adaptive search.

        Args:
            kernel_fn: The kernel function to optimize
            config_space: List of all possible configurations
            benchmark_fn: Function that takes a config and returns (latency, is_error)
            encode_fn: Function to encode config as tuple for diversity computation
            controller_config: Configuration for the adaptive controller
        """
        self.kernel_fn = kernel_fn
        self.config_space = config_space
        self.benchmark_fn = benchmark_fn
        self.encode_fn = encode_fn
        self.controller = AdaptiveController(controller_config)

        # Search state
        self._population: List[Dict[str, Any]] = []
        self._population_latencies: List[float] = []
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_latency: float = float("inf")

        # History tracking
        self._latency_history: List[float] = []
        self._time_history: List[float] = []
        self._config_history: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None

    def autotune(
        self,
        max_evaluations: int = 500,
        initial_population_size: int = 50,
        batch_size: int = 20,
        verbose: bool = True,
    ) -> AutotuneResult:
        """
        Run adaptive autotuning.

        Args:
            max_evaluations: Maximum number of config evaluations
            initial_population_size: Size of initial random population
            batch_size: Configs to evaluate per generation
            verbose: Whether to print progress

        Returns:
            AutotuneResult with best config and history
        """
        self._start_time = time.time()
        generation = 0

        if verbose:
            print("=" * 70)
            print("ADAPTIVE LFBO PATTERN SEARCH")
            print(f"Config space size: {len(self.config_space)}")
            print(f"Max evaluations: {max_evaluations}")
            print("=" * 70)

        # Phase 1: Initial random population
        if verbose:
            print(f"\n[Phase 1] Evaluating initial random population ({initial_population_size} configs)...")

        initial_configs = self._sample_random(initial_population_size)
        self._evaluate_batch(initial_configs, verbose)

        total_evaluations = len(initial_configs)
        generation = 1

        # Phase 2: Adaptive search
        if verbose:
            print(f"\n[Phase 2] Starting adaptive search...")

        while total_evaluations < max_evaluations:
            # Get current adaptive parameters
            params = self.controller.get_current_parameters()
            state = self.controller.get_current_state()

            if verbose and generation % 5 == 0:
                print(f"\n{self.controller.signal_tracker.format_status()}")
                print(f"  {self.controller.format_status()}")

            # Generate candidates based on current mode
            candidates = self._generate_candidates(
                batch_size=batch_size,
                params=params,
                state=state,
            )

            # Evaluate candidates
            latencies, errors, encodings = self._evaluate_batch(candidates, verbose=False)

            # Update controller
            self.controller.update(
                latencies=latencies,
                config_encodings=encodings,
                errors=errors,
                generation=generation,
            )

            total_evaluations += len(candidates)
            generation += 1

            # Early stopping check
            if state.improvement_velocity < 0.0001 and generation > 20:
                if state.mode == SearchMode.ESCAPE and state.population_diversity < 0.1:
                    if verbose:
                        print("\nConverged - stopping early")
                    break

        # Final results
        total_time = time.time() - self._start_time

        if verbose:
            print("\n" + "=" * 70)
            print("AUTOTUNING COMPLETE")
            print(f"Best latency: {self._best_latency:.4f} ms")
            print(f"Total evaluations: {total_evaluations}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Best config: {self._best_config}")
            print("=" * 70)

        return AutotuneResult(
            best_config=self._best_config or {},
            best_latency=self._best_latency,
            total_evaluations=total_evaluations,
            total_time_seconds=total_time,
            latency_history=self._latency_history.copy(),
            time_history=self._time_history.copy(),
            config_history=self._config_history.copy(),
            controller_history=self.controller.get_history(),
        )

    def _sample_random(self, n: int) -> List[Dict[str, Any]]:
        """Sample n random configurations."""
        if n >= len(self.config_space):
            return list(self.config_space)
        return random.sample(self.config_space, n)

    def _generate_candidates(
        self,
        batch_size: int,
        params: AdaptiveParameters,
        state: SearchState,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate configurations based on current mode.

        Args:
            batch_size: Number of candidates to generate
            params: Current adaptive parameters
            state: Current search state

        Returns:
            List of candidate configurations
        """
        candidates = []

        # Random restart in ESCAPE mode
        if state.mode == SearchMode.ESCAPE:
            num_random = int(batch_size * params.random_restart_prob)
            candidates.extend(self._sample_random(num_random))
            batch_size -= num_random

        if not self._best_config or batch_size <= 0:
            return candidates + self._sample_random(batch_size)

        # Generate neighbors around best config
        for _ in range(batch_size * 3):  # Generate 3x to allow filtering
            if len(candidates) >= batch_size:
                break

            neighbor = self._perturb_config(
                self._best_config,
                radius=params.perturbation_radius,
                jump_prob=params.cross_parameter_jump_prob,
            )

            if neighbor and neighbor not in candidates:
                candidates.append(neighbor)

        # Fill remaining with random if needed
        while len(candidates) < batch_size:
            random_config = random.choice(self.config_space)
            if random_config not in candidates:
                candidates.append(random_config)

        return candidates[:batch_size]

    def _perturb_config(
        self,
        base_config: Dict[str, Any],
        radius: int,
        jump_prob: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a perturbed config around the base.

        Args:
            base_config: Config to perturb from
            radius: Maximum perturbation distance
            jump_prob: Probability of cross-parameter jump

        Returns:
            Perturbed config or None if not possible
        """
        # Find configs within radius in config space
        base_encoding = self.encode_fn(base_config)

        candidates = []
        for config in self.config_space:
            encoding = self.encode_fn(config)

            # Compute distance
            distance = sum(a != b for a, b in zip(base_encoding, encoding))

            if 0 < distance <= radius:
                candidates.append(config)

            # Cross-parameter jump: allow larger distances occasionally
            elif distance > radius and random.random() < jump_prob:
                candidates.append(config)

        if candidates:
            return random.choice(candidates)
        return None

    def _evaluate_batch(
        self,
        configs: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Tuple[List[float], List[bool], List[Tuple[int, ...]]]:
        """
        Evaluate a batch of configurations.

        Args:
            configs: Configurations to evaluate
            verbose: Whether to print progress

        Returns:
            Tuple of (latencies, errors, encodings)
        """
        latencies = []
        errors = []
        encodings = []

        for i, config in enumerate(configs):
            try:
                latency, is_error = self.benchmark_fn(config)
            except Exception as e:
                latency = float("inf")
                is_error = True

            latencies.append(latency)
            errors.append(is_error)
            encodings.append(self.encode_fn(config))

            # Track best
            if not is_error and latency < self._best_latency:
                self._best_latency = latency
                self._best_config = config

                if verbose:
                    elapsed = time.time() - self._start_time if self._start_time else 0
                    print(f"  New best: {latency:.4f} ms (eval {len(self._latency_history) + i + 1}, {elapsed:.1f}s)")

            # Record history
            self._latency_history.append(latency if not is_error else float("inf"))
            self._time_history.append(time.time() - self._start_time if self._start_time else 0)
            self._config_history.append(config)

        return latencies, errors, encodings


class BaselineLFBOPatternSearch:
    """
    Baseline LFBO Pattern Search (non-adaptive).

    This provides a fixed-parameter search for comparison against
    the adaptive version. Parameters remain constant throughout.
    """

    def __init__(
        self,
        kernel_fn: Callable,
        config_space: List[Dict[str, Any]],
        benchmark_fn: Callable[[Dict[str, Any]], Tuple[float, bool]],
        encode_fn: Callable[[Dict[str, Any]], Tuple[int, ...]],
        perturbation_radius: int = 2,
    ):
        """
        Initialize baseline search.

        Args:
            kernel_fn: The kernel function to optimize
            config_space: List of all possible configurations
            benchmark_fn: Function that takes a config and returns (latency, is_error)
            encode_fn: Function to encode config as tuple
            perturbation_radius: Fixed perturbation radius
        """
        self.kernel_fn = kernel_fn
        self.config_space = config_space
        self.benchmark_fn = benchmark_fn
        self.encode_fn = encode_fn
        self.perturbation_radius = perturbation_radius

        self._best_config: Optional[Dict[str, Any]] = None
        self._best_latency: float = float("inf")
        self._latency_history: List[float] = []
        self._time_history: List[float] = []
        self._start_time: Optional[float] = None

    def autotune(
        self,
        max_evaluations: int = 500,
        initial_population_size: int = 50,
        batch_size: int = 20,
        verbose: bool = True,
    ) -> AutotuneResult:
        """Run baseline (non-adaptive) autotuning."""
        self._start_time = time.time()

        if verbose:
            print("=" * 70)
            print("BASELINE LFBO PATTERN SEARCH (Fixed Parameters)")
            print(f"Perturbation radius: {self.perturbation_radius}")
            print("=" * 70)

        # Initial random population
        if verbose:
            print(f"\n[Phase 1] Evaluating initial random population...")

        initial_configs = random.sample(
            self.config_space,
            min(initial_population_size, len(self.config_space))
        )

        for config in initial_configs:
            self._evaluate(config, verbose)

        total_evaluations = len(initial_configs)

        # Pattern search with fixed radius
        if verbose:
            print(f"\n[Phase 2] Starting pattern search...")

        while total_evaluations < max_evaluations:
            if not self._best_config:
                # Fallback to random if no good config found
                candidates = random.sample(
                    self.config_space,
                    min(batch_size, len(self.config_space))
                )
            else:
                # Generate neighbors with fixed radius
                candidates = self._generate_neighbors(batch_size)

            for config in candidates:
                self._evaluate(config, verbose=False)
                total_evaluations += 1

                if total_evaluations >= max_evaluations:
                    break

        total_time = time.time() - self._start_time

        if verbose:
            print("\n" + "=" * 70)
            print("BASELINE COMPLETE")
            print(f"Best latency: {self._best_latency:.4f} ms")
            print(f"Total evaluations: {total_evaluations}")
            print(f"Total time: {total_time:.2f} seconds")
            print("=" * 70)

        return AutotuneResult(
            best_config=self._best_config or {},
            best_latency=self._best_latency,
            total_evaluations=total_evaluations,
            total_time_seconds=total_time,
            latency_history=self._latency_history.copy(),
            time_history=self._time_history.copy(),
            config_history=[],
            controller_history={},
        )

    def _evaluate(self, config: Dict[str, Any], verbose: bool = True):
        """Evaluate a single configuration."""
        try:
            latency, is_error = self.benchmark_fn(config)
        except Exception:
            latency = float("inf")
            is_error = True

        if not is_error and latency < self._best_latency:
            self._best_latency = latency
            self._best_config = config
            if verbose:
                elapsed = time.time() - self._start_time if self._start_time else 0
                print(f"  New best: {latency:.4f} ms ({elapsed:.1f}s)")

        self._latency_history.append(latency if not is_error else float("inf"))
        self._time_history.append(time.time() - self._start_time if self._start_time else 0)

    def _generate_neighbors(self, n: int) -> List[Dict[str, Any]]:
        """Generate neighbors around best config."""
        if not self._best_config:
            return random.sample(self.config_space, min(n, len(self.config_space)))

        base_encoding = self.encode_fn(self._best_config)
        neighbors = []

        for config in self.config_space:
            encoding = self.encode_fn(config)
            distance = sum(a != b for a, b in zip(base_encoding, encoding))

            if 0 < distance <= self.perturbation_radius:
                neighbors.append(config)

        if len(neighbors) < n:
            # Add some random configs
            remaining = n - len(neighbors)
            neighbors.extend(random.sample(
                self.config_space,
                min(remaining, len(self.config_space))
            ))

        return random.sample(neighbors, min(n, len(neighbors)))
