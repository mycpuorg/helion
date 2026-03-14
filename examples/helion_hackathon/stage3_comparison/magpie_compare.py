"""
Magpie-Style Kernel Comparison.

Provides side-by-side evaluation of baseline vs adaptive autotuning results.
Outputs structured JSON for pipeline integration and demo visualization.
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path


@dataclass
class KernelBenchmark:
    """Benchmark result for a single kernel configuration."""
    kernel_name: str
    config: Dict[str, Any]
    latency_ms: float
    memory_bandwidth_gbps: Optional[float] = None
    correctness: bool = True
    hardware: str = "AMD MI350X"


@dataclass
class ComparisonResult:
    """Result of comparing two autotuning strategies."""
    kernel_name: str
    baseline_latency_ms: float
    adaptive_latency_ms: float
    speedup: float
    baseline_time_to_best_s: float
    adaptive_time_to_best_s: float
    time_speedup: float
    baseline_evals_to_best: int
    adaptive_evals_to_best: int
    eval_efficiency: float
    baseline_config: Dict[str, Any]
    adaptive_config: Dict[str, Any]
    hardware: str = "AMD MI350X"


@dataclass
class FullComparisonReport:
    """Complete comparison report across all kernels."""
    timestamp: str
    hardware: str
    kernels: List[ComparisonResult]
    summary: Dict[str, float]
    testcases: List[Dict[str, Any]]
    metrics: List[str]


class KernelComparator:
    """
    Magpie-style kernel comparison system.

    Compares baseline vs adaptive autotuning results across multiple kernels
    and produces structured output suitable for demo slides.

    Key metrics:
    1. Best latency achieved (absolute, in ms)
    2. Time to reach that latency (wall-clock seconds)
    3. Number of configs evaluated to reach it
    """

    def __init__(
        self,
        hardware: str = "AMD MI350X",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the comparator.

        Args:
            hardware: Target hardware name
            output_dir: Directory for output files
        """
        self.hardware = hardware
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[ComparisonResult] = []

    def compare_kernel(
        self,
        kernel_name: str,
        baseline_result: "AutotuneResult",
        adaptive_result: "AutotuneResult",
        verification_fn: Optional[Callable[[Dict, Dict], bool]] = None,
    ) -> ComparisonResult:
        """
        Compare baseline vs adaptive results for a single kernel.

        Args:
            kernel_name: Name of the kernel
            baseline_result: Result from baseline autotuning
            adaptive_result: Result from adaptive autotuning
            verification_fn: Optional function to verify correctness

        Returns:
            ComparisonResult with all metrics
        """
        # Compute speedup
        baseline_latency = baseline_result.best_latency
        adaptive_latency = adaptive_result.best_latency

        if baseline_latency > 0 and baseline_latency != float("inf"):
            speedup = baseline_latency / adaptive_latency
        else:
            speedup = 1.0

        # Find time and evaluations to reach best
        baseline_time_to_best = self._find_time_to_best(baseline_result)
        adaptive_time_to_best = self._find_time_to_best(adaptive_result)

        if baseline_time_to_best > 0:
            time_speedup = baseline_time_to_best / adaptive_time_to_best
        else:
            time_speedup = 1.0

        baseline_evals = self._find_evals_to_best(baseline_result)
        adaptive_evals = self._find_evals_to_best(adaptive_result)

        if baseline_evals > 0:
            eval_efficiency = baseline_evals / adaptive_evals
        else:
            eval_efficiency = 1.0

        result = ComparisonResult(
            kernel_name=kernel_name,
            baseline_latency_ms=baseline_latency,
            adaptive_latency_ms=adaptive_latency,
            speedup=speedup,
            baseline_time_to_best_s=baseline_time_to_best,
            adaptive_time_to_best_s=adaptive_time_to_best,
            time_speedup=time_speedup,
            baseline_evals_to_best=baseline_evals,
            adaptive_evals_to_best=adaptive_evals,
            eval_efficiency=eval_efficiency,
            baseline_config=baseline_result.best_config,
            adaptive_config=adaptive_result.best_config,
            hardware=self.hardware,
        )

        self._results.append(result)
        return result

    def _find_time_to_best(self, result: "AutotuneResult") -> float:
        """Find wall-clock time when best latency was first achieved."""
        best = result.best_latency
        threshold = best * 1.05  # Within 5% of best

        for i, (lat, t) in enumerate(zip(result.latency_history, result.time_history)):
            if lat <= threshold:
                return t

        return result.total_time_seconds

    def _find_evals_to_best(self, result: "AutotuneResult") -> int:
        """Find number of evaluations when best latency was achieved."""
        best = result.best_latency
        threshold = best * 1.05  # Within 5% of best

        for i, lat in enumerate(result.latency_history):
            if lat <= threshold:
                return i + 1

        return result.total_evaluations

    def generate_report(
        self,
        testcases: Optional[List[Dict[str, Any]]] = None,
    ) -> FullComparisonReport:
        """
        Generate a full comparison report.

        Args:
            testcases: Optional list of test case configurations

        Returns:
            FullComparisonReport with all results
        """
        if testcases is None:
            testcases = [
                {"shape": [2048, 4096]},
                {"shape": [4096, 8192]},
            ]

        # Compute summary statistics
        if self._results:
            avg_speedup = sum(r.speedup for r in self._results) / len(self._results)
            avg_time_speedup = sum(r.time_speedup for r in self._results) / len(self._results)
            avg_eval_efficiency = sum(r.eval_efficiency for r in self._results) / len(self._results)
            max_speedup = max(r.speedup for r in self._results)
        else:
            avg_speedup = avg_time_speedup = avg_eval_efficiency = max_speedup = 1.0

        summary = {
            "avg_latency_speedup": avg_speedup,
            "avg_time_to_best_speedup": avg_time_speedup,
            "avg_eval_efficiency": avg_eval_efficiency,
            "max_latency_speedup": max_speedup,
            "num_kernels": len(self._results),
        }

        return FullComparisonReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware=self.hardware,
            kernels=self._results.copy(),
            summary=summary,
            testcases=testcases,
            metrics=["latency_ms", "memory_bandwidth_utilization", "correctness"],
        )

    def to_json(self, report: Optional[FullComparisonReport] = None) -> str:
        """Convert report to JSON string."""
        if report is None:
            report = self.generate_report()

        return json.dumps({
            "timestamp": report.timestamp,
            "hardware": report.hardware,
            "kernels": [asdict(k) for k in report.kernels],
            "summary": report.summary,
            "testcases": report.testcases,
            "metrics": report.metrics,
        }, indent=2)

    def save_report(self, filename: str = "comparison_report.json"):
        """Save report to file."""
        report = self.generate_report()
        output_path = self.output_dir / filename
        output_path.write_text(self.to_json(report))
        return output_path

    def format_comparison_table(self) -> str:
        """Format results as a readable comparison table."""
        lines = []
        lines.append("=" * 120)
        lines.append("SIDE-BY-SIDE COMPARISON: Baseline LFBO vs Adaptive EvoX Strategy")
        lines.append(f"Hardware: {self.hardware}")
        lines.append("=" * 120)
        lines.append("")

        # Header
        header = (
            f"{'Kernel':<20} "
            f"{'Baseline (ms)':<14} "
            f"{'Adaptive (ms)':<14} "
            f"{'Speedup':<10} "
            f"{'Base Time (s)':<14} "
            f"{'Adapt Time (s)':<14} "
            f"{'Time Speedup':<12}"
        )
        lines.append(header)
        lines.append("-" * 120)

        for r in self._results:
            line = (
                f"{r.kernel_name:<20} "
                f"{r.baseline_latency_ms:<14.4f} "
                f"{r.adaptive_latency_ms:<14.4f} "
                f"{r.speedup:<10.2f}x "
                f"{r.baseline_time_to_best_s:<14.2f} "
                f"{r.adaptive_time_to_best_s:<14.2f} "
                f"{r.time_speedup:<12.2f}x"
            )
            lines.append(line)

        lines.append("-" * 120)

        # Summary
        report = self.generate_report()
        lines.append("")
        lines.append("SUMMARY:")
        lines.append(f"  Average latency speedup: {report.summary['avg_latency_speedup']:.2f}x")
        lines.append(f"  Average time-to-best speedup: {report.summary['avg_time_to_best_speedup']:.2f}x")
        lines.append(f"  Average evaluation efficiency: {report.summary['avg_eval_efficiency']:.2f}x")
        lines.append("")
        lines.append("=" * 120)

        return "\n".join(lines)


def compare_autotuning_runs(
    kernel_name: str,
    baseline_result: "AutotuneResult",
    adaptive_result: "AutotuneResult",
    hardware: str = "AMD MI350X",
) -> Tuple[ComparisonResult, str]:
    """
    Convenience function to compare two autotuning runs.

    Args:
        kernel_name: Name of the kernel
        baseline_result: Baseline autotuning result
        adaptive_result: Adaptive autotuning result
        hardware: Target hardware

    Returns:
        Tuple of (ComparisonResult, formatted_table)
    """
    comparator = KernelComparator(hardware=hardware)
    result = comparator.compare_kernel(kernel_name, baseline_result, adaptive_result)
    table = comparator.format_comparison_table()
    return result, table
