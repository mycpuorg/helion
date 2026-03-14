# Stage 3: Magpie Comparison and Evaluation
# Side-by-side evaluation of baseline vs adaptive autotuning

from .magpie_compare import KernelComparator, ComparisonResult
from .results_formatter import ResultsFormatter, DemoSlideGenerator

__all__ = [
    "KernelComparator",
    "ComparisonResult",
    "ResultsFormatter",
    "DemoSlideGenerator",
]
