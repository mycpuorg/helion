# Stage 2: EvoX-Style Adaptive Autotuning
# Implements adaptive strategy switching based on search state signals
#
# Two implementations:
#   1. adaptive_lfbo_search.py — REAL integration: subclasses Helion's
#      LFBOPatternSearch and modulates its actual parameters per generation.
#      This is what should be used for the hackathon demo.
#
#   2. lfbo_wrapper.py — Standalone simulation (original prototype).
#      Useful for offline testing without GPU/Helion dependencies, but
#      does NOT use Helion's real infrastructure.

# Real Helion integration (preferred)
from .adaptive_lfbo_search import AdaptiveLFBOPatternSearch
from .adaptive_lfbo_search import AdaptiveConfig
from .adaptive_lfbo_search import SearchMode
from .adaptive_lfbo_search import SignalTracker

# Standalone prototype (for reference / offline testing)
from .adaptive_controller import AdaptiveController, ControllerConfig
from .lfbo_wrapper import AdaptiveLFBOPatternSearch as StandaloneAdaptiveSearch
from .lfbo_wrapper import BaselineLFBOPatternSearch

__all__ = [
    # Real integration
    "AdaptiveLFBOPatternSearch",
    "AdaptiveConfig",
    "SearchMode",
    "SignalTracker",
    # Standalone reference
    "AdaptiveController",
    "ControllerConfig",
    "StandaloneAdaptiveSearch",
    "BaselineLFBOPatternSearch",
]
