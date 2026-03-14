"""
Adaptive Controller for EvoX-Style Autotuning.

Implements the state machine meta-controller that adjusts search parameters
based on observed signals. This is the core innovation that applies
EvoX principles to Helion's LFBO pattern search.

The controller adjusts:
1. Perturbation radius (how far to search from current best)
2. Classifier confidence threshold (how selective to be)
3. Diversity penalty (how much to penalize similar configs)
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import math

from .signals import SignalTracker, SearchState, SearchMode, SignalConfig


@dataclass
class ControllerConfig:
    """Configuration for the adaptive controller."""
    # Base parameters (will be modulated by mode)
    base_perturbation_radius: int = 2
    base_classifier_threshold: float = 0.3  # Quantile for "good" configs
    base_diversity_penalty: float = 1.0

    # Mode-specific multipliers
    exploit_radius_multiplier: float = 0.5    # Tighter search in EXPLOIT
    exploit_threshold_multiplier: float = 1.2  # Higher confidence required
    explore_radius_multiplier: float = 1.5     # Wider search in EXPLORE
    explore_threshold_multiplier: float = 0.8  # Lower confidence accepted
    escape_radius_multiplier: float = 3.0      # Very wide in ESCAPE
    escape_threshold_multiplier: float = 0.5   # Accept more varied configs

    # Transition smoothing
    smoothing_factor: float = 0.3  # EMA factor for parameter updates

    # Escape mode settings
    escape_random_restart_prob: float = 0.5
    escape_cross_parameter_jump_prob: float = 0.3


@dataclass
class AdaptiveParameters:
    """Current adaptive parameters for the search."""
    perturbation_radius: int = 2
    classifier_threshold: float = 0.3
    diversity_penalty: float = 1.0
    random_restart_prob: float = 0.0
    cross_parameter_jump_prob: float = 0.0


class AdaptiveController:
    """
    State machine meta-controller for adaptive autotuning.

    This controller observes the search state through signals and
    adjusts search parameters to optimize the exploration/exploitation
    tradeoff dynamically.

    The key insight from EvoX is that the selection mechanism itself
    should adapt based on search progress, not just use fixed strategies.
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        signal_config: Optional[SignalConfig] = None,
    ):
        """
        Initialize the adaptive controller.

        Args:
            config: Controller configuration
            signal_config: Signal tracker configuration
        """
        self.config = config or ControllerConfig()
        self.signal_tracker = SignalTracker(signal_config)

        # Current parameters (smoothed)
        self._current_params = AdaptiveParameters(
            perturbation_radius=self.config.base_perturbation_radius,
            classifier_threshold=self.config.base_classifier_threshold,
            diversity_penalty=self.config.base_diversity_penalty,
        )

        # History for analysis
        self._state_history: List[SearchState] = []
        self._param_history: List[AdaptiveParameters] = []

    def update(
        self,
        latencies: List[float],
        config_encodings: List[tuple],
        errors: List[bool],
        generation: int,
    ) -> AdaptiveParameters:
        """
        Update the controller with new evaluation results.

        Args:
            latencies: Latencies from recent evaluations
            config_encodings: Encoded configurations
            errors: Whether each evaluation was an error
            generation: Current generation number

        Returns:
            Updated adaptive parameters
        """
        # Record observations
        self.signal_tracker.set_generation(generation)
        for lat, enc, err in zip(latencies, config_encodings, errors):
            self.signal_tracker.record_evaluation(lat, enc, err)

        # Get current state
        state = self.signal_tracker.get_current_state()
        self._state_history.append(state)

        # Compute target parameters for current mode
        target_params = self._compute_target_parameters(state)

        # Smooth transition to target
        self._current_params = self._smooth_transition(
            self._current_params,
            target_params,
        )
        self._param_history.append(self._current_params)

        return self._current_params

    def get_current_parameters(self) -> AdaptiveParameters:
        """Get the current adaptive parameters."""
        return self._current_params

    def get_current_state(self) -> SearchState:
        """Get the current search state with signals."""
        return self.signal_tracker.get_current_state()

    def _compute_target_parameters(self, state: SearchState) -> AdaptiveParameters:
        """
        Compute target parameters for the current mode.

        Args:
            state: Current search state

        Returns:
            Target parameters for this mode
        """
        cfg = self.config

        if state.mode == SearchMode.EXPLOIT:
            # Focus on local refinement
            radius = max(1, int(cfg.base_perturbation_radius * cfg.exploit_radius_multiplier))
            threshold = cfg.base_classifier_threshold * cfg.exploit_threshold_multiplier
            diversity = cfg.base_diversity_penalty * 0.5  # Less diversity needed
            restart_prob = 0.0
            jump_prob = 0.0

        elif state.mode == SearchMode.EXPLORE:
            # Broaden search
            radius = max(1, int(cfg.base_perturbation_radius * cfg.explore_radius_multiplier))
            threshold = cfg.base_classifier_threshold * cfg.explore_threshold_multiplier
            diversity = cfg.base_diversity_penalty * 1.5  # More diversity wanted
            restart_prob = 0.0
            jump_prob = 0.1  # Small chance of cross-parameter jumps

        else:  # ESCAPE
            # Force exploration of new regions
            radius = max(1, int(cfg.base_perturbation_radius * cfg.escape_radius_multiplier))
            threshold = cfg.base_classifier_threshold * cfg.escape_threshold_multiplier
            diversity = cfg.base_diversity_penalty * 2.0  # Maximize diversity
            restart_prob = cfg.escape_random_restart_prob
            jump_prob = cfg.escape_cross_parameter_jump_prob

        return AdaptiveParameters(
            perturbation_radius=radius,
            classifier_threshold=min(0.5, max(0.1, threshold)),
            diversity_penalty=diversity,
            random_restart_prob=restart_prob,
            cross_parameter_jump_prob=jump_prob,
        )

    def _smooth_transition(
        self,
        current: AdaptiveParameters,
        target: AdaptiveParameters,
    ) -> AdaptiveParameters:
        """
        Smoothly transition from current to target parameters.

        Uses exponential moving average for smooth transitions.

        Args:
            current: Current parameters
            target: Target parameters

        Returns:
            Smoothed parameters
        """
        alpha = self.config.smoothing_factor

        return AdaptiveParameters(
            perturbation_radius=round(
                current.perturbation_radius * (1 - alpha) +
                target.perturbation_radius * alpha
            ),
            classifier_threshold=(
                current.classifier_threshold * (1 - alpha) +
                target.classifier_threshold * alpha
            ),
            diversity_penalty=(
                current.diversity_penalty * (1 - alpha) +
                target.diversity_penalty * alpha
            ),
            random_restart_prob=(
                current.random_restart_prob * (1 - alpha) +
                target.random_restart_prob * alpha
            ),
            cross_parameter_jump_prob=(
                current.cross_parameter_jump_prob * (1 - alpha) +
                target.cross_parameter_jump_prob * alpha
            ),
        )

    def format_status(self) -> str:
        """Format current controller status."""
        state = self.get_current_state()
        params = self._current_params
        return (
            f"Mode: {state.mode.value} | "
            f"Radius: {params.perturbation_radius} | "
            f"Threshold: {params.classifier_threshold:.2f} | "
            f"Diversity: {params.diversity_penalty:.2f}"
        )

    def get_history(self) -> Dict[str, List[Any]]:
        """Get history of states and parameters for analysis."""
        return {
            "states": [
                {
                    "mode": s.mode.value,
                    "velocity": s.improvement_velocity,
                    "diversity": s.population_diversity,
                    "error": s.error_pressure,
                    "generation": s.generation,
                    "best_latency": s.best_latency,
                }
                for s in self._state_history
            ],
            "parameters": [
                {
                    "radius": p.perturbation_radius,
                    "threshold": p.classifier_threshold,
                    "diversity_penalty": p.diversity_penalty,
                    "restart_prob": p.random_restart_prob,
                    "jump_prob": p.cross_parameter_jump_prob,
                }
                for p in self._param_history
            ],
        }
