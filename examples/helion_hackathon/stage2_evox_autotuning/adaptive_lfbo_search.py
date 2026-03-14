"""
Adaptive LFBO Pattern Search — EvoX-style strategy switching for Helion's autotuner.

This module subclasses the real LFBOPatternSearch and adds an adaptive
meta-controller that modulates search parameters per generation based on
three observable signals:

  1. improvement_velocity — rolling rate of best-perf improvement
  2. population_diversity — mean leaf-similarity across recent candidates
  3. error_pressure — fraction of inf-perf (compile fail / timeout) configs

These drive a three-mode state machine:
  EXPLOIT — tighten radius, raise selectivity (things are improving)
  EXPLORE — widen radius, relax selectivity (stagnant but diverse)
  ESCAPE  — maximum radius, inject random restarts (stuck in local opt)

Integration points with the real LFBO:
  self.radius            → perturbation distance in config space
  self.quantile          → classifier threshold for good/bad split
  self.similarity_penalty → diversity penalty coefficient
  self.num_neighbors     → number of random neighbors per generation
  self.frac_selected     → fraction of candidates to benchmark
  self.patience          → early-stopping patience counter

Usage:
  Register "AdaptiveLFBOPatternSearch" in helion/autotuner/__init__.py's
  search_algorithms dict, then select it via:
    settings.autotune_search_algorithm = "AdaptiveLFBOPatternSearch"

  Or use directly:
    AdaptiveLFBOPatternSearch(kernel, args, ...).autotune()
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.autotuner.base_search import _AutotunableKernel
    from helion.runtime.config import Config


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


class SearchMode(enum.Enum):
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    ESCAPE = "escape"


@dataclass
class AdaptiveSignals:
    """Snapshot of the three observable signals at one generation."""

    mode: SearchMode = SearchMode.EXPLORE
    improvement_velocity: float = 0.0
    population_diversity: float = 1.0
    error_pressure: float = 0.0
    generation: int = 0
    best_perf: float = math.inf


@dataclass
class AdaptiveConfig:
    """Thresholds that govern mode transitions."""

    # Velocity thresholds (relative improvement per generation)
    high_velocity: float = 0.01  # >1 % improvement → EXPLOIT
    low_velocity: float = 0.001  # <0.1 % improvement → check diversity

    # Diversity thresholds (0–1 scale from leaf-similarity)
    high_diversity: float = 0.6
    low_diversity: float = 0.2

    # Error pressure
    high_error: float = 0.3  # >30 % failures → ESCAPE

    # Parameter multipliers per mode (applied to base values)
    exploit_radius_mult: float = 0.5
    exploit_quantile_mult: float = 0.8  # tighter good/bad threshold
    exploit_similarity_mult: float = 0.5
    exploit_neighbors_mult: float = 0.7
    exploit_frac_selected_mult: float = 1.5  # benchmark more of the promising ones

    explore_radius_mult: float = 1.5
    explore_quantile_mult: float = 1.2
    explore_similarity_mult: float = 1.5
    explore_neighbors_mult: float = 1.3
    explore_frac_selected_mult: float = 1.0

    escape_radius_mult: float = 3.0
    escape_quantile_mult: float = 1.5  # very loose threshold
    escape_similarity_mult: float = 2.0
    escape_neighbors_mult: float = 1.5
    escape_frac_selected_mult: float = 0.8

    # EMA smoothing factor for parameter transitions
    smoothing: float = 0.4

    # Rolling window sizes
    velocity_window: int = 5  # generations
    error_window: int = 30  # individual configs


class SignalTracker:
    """Tracks signals across generations using real autotuner data."""

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self.config = config or AdaptiveConfig()
        self._best_perfs: deque[float] = deque(
            maxlen=self.config.velocity_window + 1
        )
        self._error_history: deque[bool] = deque(maxlen=self.config.error_window)
        self._last_diversity: float = 1.0
        self._generation = 0

    def record_generation(
        self,
        best_perf: float,
        population: list[PopulationMember],
        mean_leaf_similarity: float | None = None,
    ) -> AdaptiveSignals:
        """Record one generation's results and compute signals."""
        self._generation += 1

        # Track best perf history
        if math.isfinite(best_perf):
            self._best_perfs.append(best_perf)

        # Error pressure from population
        for member in population:
            self._error_history.append(not math.isfinite(member.perf))

        # Compute signals
        velocity = self._compute_velocity()
        diversity = (
            mean_leaf_similarity if mean_leaf_similarity is not None else 1.0
        )
        # Invert: high similarity → low diversity
        diversity = max(0.0, 1.0 - diversity)
        self._last_diversity = diversity
        error = self._compute_error_pressure()

        mode = self._determine_mode(velocity, diversity, error)

        return AdaptiveSignals(
            mode=mode,
            improvement_velocity=velocity,
            population_diversity=diversity,
            error_pressure=error,
            generation=self._generation,
            best_perf=best_perf,
        )

    def _compute_velocity(self) -> float:
        if len(self._best_perfs) < 2:
            return 0.0
        perfs = list(self._best_perfs)
        mid = len(perfs) // 2
        old_best = min(perfs[:mid]) if mid > 0 else perfs[0]
        new_best = min(perfs[mid:])
        if old_best <= 0 or not math.isfinite(old_best):
            return 0.0
        improvement = (old_best - new_best) / old_best
        return max(0.0, improvement / max(1, len(perfs) - mid))

    def _compute_error_pressure(self) -> float:
        if not self._error_history:
            return 0.0
        return sum(self._error_history) / len(self._error_history)

    def _determine_mode(
        self, velocity: float, diversity: float, error: float
    ) -> SearchMode:
        cfg = self.config
        if error > cfg.high_error:
            return SearchMode.ESCAPE
        if velocity > cfg.high_velocity:
            return SearchMode.EXPLOIT
        if velocity < cfg.low_velocity:
            if diversity < cfg.low_diversity:
                return SearchMode.ESCAPE
            return SearchMode.EXPLORE
        # Medium velocity — slight preference for exploit
        return SearchMode.EXPLOIT


# ---------------------------------------------------------------------------
# Adaptive LFBO Pattern Search
# ---------------------------------------------------------------------------


class AdaptiveLFBOPatternSearch(LFBOPatternSearch):
    """
    LFBOPatternSearch with EvoX-style adaptive parameter control.

    Monitors improvement velocity, population diversity, and error pressure
    per generation, then adjusts radius, quantile, similarity_penalty,
    num_neighbors, and frac_selected to balance exploration vs exploitation.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        # Standard LFBO parameters (serve as BASE values for adaptation)
        initial_population: int = 100,
        copies: int = 5,
        max_generations: int = 20,
        min_improvement_delta: float = 0.001,
        frac_selected: float = 0.10,
        num_neighbors: int = 300,
        radius: int = 2,
        quantile: float = 0.1,
        patience: int = 1,
        similarity_penalty: float = 1.0,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        compile_timeout_lower_bound: float = 30.0,
        compile_timeout_quantile: float = 0.9,
        # Adaptive-specific config
        adaptive_config: AdaptiveConfig | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel,
            args=args,
            initial_population=initial_population,
            copies=copies,
            max_generations=max_generations,
            min_improvement_delta=min_improvement_delta,
            frac_selected=frac_selected,
            num_neighbors=num_neighbors,
            radius=radius,
            quantile=quantile,
            patience=patience,
            similarity_penalty=similarity_penalty,
            initial_population_strategy=initial_population_strategy,
            compile_timeout_lower_bound=compile_timeout_lower_bound,
            compile_timeout_quantile=compile_timeout_quantile,
        )

        # Store base values for modulation
        self._base_radius = radius
        self._base_quantile = quantile
        self._base_similarity_penalty = similarity_penalty
        self._base_num_neighbors = num_neighbors
        self._base_frac_selected = frac_selected

        # Adaptive controller
        self._adaptive_config = adaptive_config or AdaptiveConfig()
        self._signal_tracker = SignalTracker(self._adaptive_config)
        self._signal_history: list[AdaptiveSignals] = []

        # Smoothed current values (start at base)
        self._smooth_radius = float(radius)
        self._smooth_quantile = quantile
        self._smooth_similarity = similarity_penalty
        self._smooth_neighbors = float(num_neighbors)
        self._smooth_frac = frac_selected

    def _adapt_parameters(self, signals: AdaptiveSignals) -> None:
        """Compute target parameters for current mode and smooth toward them."""
        cfg = self._adaptive_config
        alpha = cfg.smoothing

        if signals.mode == SearchMode.EXPLOIT:
            target_radius = self._base_radius * cfg.exploit_radius_mult
            target_quantile = self._base_quantile * cfg.exploit_quantile_mult
            target_sim = self._base_similarity_penalty * cfg.exploit_similarity_mult
            target_neighbors = self._base_num_neighbors * cfg.exploit_neighbors_mult
            target_frac = self._base_frac_selected * cfg.exploit_frac_selected_mult
        elif signals.mode == SearchMode.EXPLORE:
            target_radius = self._base_radius * cfg.explore_radius_mult
            target_quantile = self._base_quantile * cfg.explore_quantile_mult
            target_sim = self._base_similarity_penalty * cfg.explore_similarity_mult
            target_neighbors = self._base_num_neighbors * cfg.explore_neighbors_mult
            target_frac = self._base_frac_selected * cfg.explore_frac_selected_mult
        else:  # ESCAPE
            target_radius = self._base_radius * cfg.escape_radius_mult
            target_quantile = self._base_quantile * cfg.escape_quantile_mult
            target_sim = self._base_similarity_penalty * cfg.escape_similarity_mult
            target_neighbors = self._base_num_neighbors * cfg.escape_neighbors_mult
            target_frac = self._base_frac_selected * cfg.escape_frac_selected_mult

        # EMA smoothing
        self._smooth_radius = self._smooth_radius * (1 - alpha) + target_radius * alpha
        self._smooth_quantile = (
            self._smooth_quantile * (1 - alpha) + target_quantile * alpha
        )
        self._smooth_similarity = (
            self._smooth_similarity * (1 - alpha) + target_sim * alpha
        )
        self._smooth_neighbors = (
            self._smooth_neighbors * (1 - alpha) + target_neighbors * alpha
        )
        self._smooth_frac = self._smooth_frac * (1 - alpha) + target_frac * alpha

        # Apply to real LFBO parameters
        self.radius = max(1, round(self._smooth_radius))
        self.quantile = max(0.05, min(0.5, self._smooth_quantile))
        self.similarity_penalty = max(0.0, self._smooth_similarity)
        self.num_neighbors = max(50, round(self._smooth_neighbors))
        self.frac_selected = max(0.05, min(0.5, self._smooth_frac))

    def _autotune(self) -> Config:
        """Override _autotune to inject adaptive parameter control per generation."""
        from helion.autotuner.base_search import performance

        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting AdaptiveLFBOPatternSearch with initial_population={initial_population_name},"
            f" copies={self.copies},"
            f" max_generations={self.max_generations},"
            f" similarity_penalty={self.similarity_penalty}"
            f" (adaptive mode enabled)"
        )
        visited: set[Config] = set()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member.config not in visited:
                visited.add(member.config)
                self.population.append(member)
        self.set_generation(0)
        self.parallel_benchmark_population(self.population, desc="Initial population")

        # Compute adaptive compile timeout
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # Rebenchmark for accuracy
        self.rebenchmark_population(self.population, desc="Verifying initial results")
        self.population.sort(key=performance)

        starting_points = [
            m for m in self.population[: self.copies] if math.isfinite(m.perf)
        ]
        self.log(
            f"Initial random population of {len(self.population)}, "
            f"{len(starting_points)} starting points:",
            self.statistics,
        )
        if not starting_points:
            from helion import exc

            raise exc.NoConfigFound

        # Save initial training data
        for member in self.population:
            self.train_x.append(self.config_gen.encode_config(member.flat_values))
            self.train_y.append(member.perf)

        self._fit_surrogate()

        search_copies = [
            self._pruned_pattern_search_from(m, visited) for m in starting_points
        ]

        for generation in range(1, self.max_generations + 1):
            prior_best = self.best
            new_population = {id(prior_best): prior_best}
            num_neighbors = 0
            num_active = 0
            for search_copy in search_copies:
                added = next(search_copy, ())
                if added:
                    assert len(added) > 1
                    num_active += 1
                    num_neighbors += len(added) - 1
                    for member in added:
                        new_population[id(member)] = member
            if num_active == 0:
                break

            self.log(
                f"Generation {generation} starting: {num_neighbors} neighbors, "
                f"{num_active} active search path(s)"
            )

            self.population = [*new_population.values()]
            unbenchmarked = [m for m in self.population if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(generation)
                self.parallel_benchmark_population(
                    unbenchmarked, desc=f"Generation {generation}:"
                )
            self.rebenchmark_population(
                self.population, desc=f"Generation {generation}: verifying top configs"
            )
            self.log(f"Generation {generation} complete:", self.statistics)

            # Update training data
            for member in self.population:
                self.train_x.append(self.config_gen.encode_config(member.flat_values))
                self.train_y.append(member.perf)

            self._fit_surrogate()

            # ---- ADAPTIVE CONTROL: compute signals and adjust parameters ----
            mean_sim = self._estimate_population_similarity()
            signals = self._signal_tracker.record_generation(
                best_perf=self.best.perf,
                population=self.population,
                mean_leaf_similarity=mean_sim,
            )
            self._signal_history.append(signals)
            self._adapt_parameters(signals)

            self.log(
                f"[Adaptive] mode={signals.mode.value}, "
                f"velocity={signals.improvement_velocity:.4f}, "
                f"diversity={signals.population_diversity:.3f}, "
                f"errors={signals.error_pressure:.2%} → "
                f"radius={self.radius}, quantile={self.quantile:.3f}, "
                f"sim_penalty={self.similarity_penalty:.2f}, "
                f"neighbors={self.num_neighbors}, frac={self.frac_selected:.3f}"
            )

        best = self.run_finishing_phase(self.best, self.finishing_rounds)
        return best.config

    def _estimate_population_similarity(self) -> float | None:
        """Estimate mean pairwise leaf similarity across recent population.

        Uses the fitted surrogate's leaf-node co-occurrence if available.
        Returns None if no surrogate is fitted yet.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        surrogate = self.surrogate
        if surrogate is None:
            return None

        # Encode recent population members
        members = [m for m in self.population if math.isfinite(m.perf)]
        if len(members) < 2:
            return None

        X = np.array(
            [self.config_gen.encode_config(m.flat_values) for m in members]
        )

        # Use the existing leaf_similarity method from LFBOPatternSearch
        sim_matrix = self.compute_leaf_similarity(surrogate, X)

        # Mean off-diagonal similarity
        n = sim_matrix.shape[0]
        total = sim_matrix.sum() - np.trace(sim_matrix)
        pairs = n * (n - 1)
        return float(total / pairs) if pairs > 0 else None
