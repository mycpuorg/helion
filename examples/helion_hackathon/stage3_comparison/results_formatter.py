"""
Results Formatter and Demo Slide Generator.

Formats comparison results for:
1. Demo slides (Markdown tables)
2. Latency-vs-time curves
3. Summary reports
"""
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


class ResultsFormatter:
    """
    Formats autotuning results for various output formats.
    """

    @staticmethod
    def format_kernel_selection_slide(selection_results: List[Dict]) -> str:
        """
        Format Stage 1 results for demo Slide 1.

        Args:
            selection_results: List of kernel selection results

        Returns:
            Markdown formatted slide content
        """
        lines = []
        lines.append("# Slide 1: Kernel Selection (Magpie)")
        lines.append("")
        lines.append("## Kernels Selected for Autotuning")
        lines.append("")
        lines.append("| Kernel | Wall-time % | Config Space | Selection Score |")
        lines.append("|--------|-------------|--------------|-----------------|")

        total_wall_time = 0
        for r in selection_results:
            space_str = f"{r['config_space_size']:.1e}"
            lines.append(
                f"| {r['kernel_name']} | {r['wall_time_fraction']*100:.1f}% | "
                f"{space_str} | {r['selection_score']:.3f} |"
            )
            total_wall_time += r['wall_time_fraction']

        lines.append("")
        lines.append(f"**Bottom line:** These {len(selection_results)} kernels represent "
                     f"{total_wall_time*100:.1f}% of model execution time")
        lines.append("")
        lines.append("*Selection formula: score = wall_time_fraction × log(config_space_size)*")

        return "\n".join(lines)

    @staticmethod
    def format_autotuning_slide(comparison_results: List[Dict]) -> str:
        """
        Format Stage 2/3 results for demo Slide 2.

        Args:
            comparison_results: List of comparison results

        Returns:
            Markdown formatted slide content
        """
        lines = []
        lines.append("# Slide 2: Adaptive Autotuning (EvoX + Helion)")
        lines.append("")
        lines.append("## Latency Improvements")
        lines.append("")
        lines.append("| Kernel | Baseline (ms) | Adaptive (ms) | Speedup |")
        lines.append("|--------|---------------|---------------|---------|")

        for r in comparison_results:
            lines.append(
                f"| {r['kernel_name']} | {r['baseline_latency_ms']:.4f} | "
                f"{r['adaptive_latency_ms']:.4f} | {r['speedup']:.2f}x |"
            )

        lines.append("")
        lines.append("## Time-to-Best Performance")
        lines.append("")
        lines.append("| Kernel | Baseline Time (s) | Adaptive Time (s) | Time Speedup |")
        lines.append("|--------|-------------------|-------------------|--------------|")

        for r in comparison_results:
            lines.append(
                f"| {r['kernel_name']} | {r['baseline_time_to_best_s']:.1f} | "
                f"{r['adaptive_time_to_best_s']:.1f} | {r['time_speedup']:.2f}x |"
            )

        lines.append("")
        lines.append("**Key callout:** Adaptive strategy converges to within 5% of best "
                     "latency faster than baseline")

        return "\n".join(lines)

    @staticmethod
    def format_comparison_slide(report: Dict) -> str:
        """
        Format final comparison for demo Slide 3.

        Args:
            report: Full comparison report dict

        Returns:
            Markdown formatted slide content
        """
        lines = []
        lines.append("# Slide 3: Side-by-Side Results (Magpie Compare)")
        lines.append("")
        lines.append(f"**Hardware:** {report.get('hardware', 'AMD MI350X')}")
        lines.append("")
        lines.append("## Final Comparison")
        lines.append("")
        lines.append("| Kernel | Best Latency (ms) | Improvement % | Eval Efficiency |")
        lines.append("|--------|-------------------|---------------|-----------------|")

        for k in report.get('kernels', []):
            improvement = (1 - k['adaptive_latency_ms'] / k['baseline_latency_ms']) * 100
            lines.append(
                f"| {k['kernel_name']} | {k['adaptive_latency_ms']:.4f} | "
                f"{improvement:.1f}% | {k['eval_efficiency']:.2f}x |"
            )

        lines.append("")
        lines.append("## Summary")
        summary = report.get('summary', {})
        lines.append(f"- **Average latency speedup:** {summary.get('avg_latency_speedup', 1):.2f}x")
        lines.append(f"- **Average time-to-best speedup:** {summary.get('avg_time_to_best_speedup', 1):.2f}x")
        lines.append(f"- **Average evaluation efficiency:** {summary.get('avg_eval_efficiency', 1):.2f}x")

        return "\n".join(lines)


class DemoSlideGenerator:
    """
    Generates complete demo slide deck from pipeline results.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formatter = ResultsFormatter()

    def generate_slides(
        self,
        selection_results: List[Dict],
        comparison_results: List[Dict],
        full_report: Dict,
    ) -> str:
        """
        Generate complete demo slides.

        Args:
            selection_results: Stage 1 kernel selection results
            comparison_results: Stage 2/3 comparison results
            full_report: Full comparison report

        Returns:
            Complete markdown content for all slides
        """
        slides = []

        # Title slide
        slides.append("# EvoX-Helion + Magpie: Adaptive Autotuning for AMD MI350X")
        slides.append("")
        slides.append("*\"We used Magpie to find which kernels matter most in real AMD workloads, ")
        slides.append("applied EvoX-style adaptive autotuning to Helion on those kernels, ")
        slides.append("and showed side-by-side latency speedups on MI350X.\"*")
        slides.append("")
        slides.append("---")
        slides.append("")

        # Slide 1: Kernel Selection
        slides.append(self.formatter.format_kernel_selection_slide(selection_results))
        slides.append("")
        slides.append("---")
        slides.append("")

        # Slide 2: Autotuning Results
        slides.append(self.formatter.format_autotuning_slide(comparison_results))
        slides.append("")
        slides.append("---")
        slides.append("")

        # Slide 3: Final Comparison
        slides.append(self.formatter.format_comparison_slide(full_report))

        return "\n".join(slides)

    def save_slides(
        self,
        selection_results: List[Dict],
        comparison_results: List[Dict],
        full_report: Dict,
        filename: str = "demo_slides.md",
    ) -> Path:
        """
        Generate and save demo slides.

        Returns:
            Path to the saved file
        """
        content = self.generate_slides(selection_results, comparison_results, full_report)
        output_path = self.output_dir / filename
        output_path.write_text(content)
        return output_path

    def generate_latency_plot_data(
        self,
        baseline_history: List[float],
        baseline_times: List[float],
        adaptive_history: List[float],
        adaptive_times: List[float],
        kernel_name: str,
    ) -> Dict[str, Any]:
        """
        Generate data for latency-vs-time plot.

        Returns:
            Dict with plot data in a format suitable for matplotlib
        """
        # Compute running minimum (best-so-far)
        baseline_best = []
        adaptive_best = []
        current_baseline = float("inf")
        current_adaptive = float("inf")

        for lat in baseline_history:
            if lat < current_baseline:
                current_baseline = lat
            baseline_best.append(current_baseline)

        for lat in adaptive_history:
            if lat < current_adaptive:
                current_adaptive = lat
            adaptive_best.append(current_adaptive)

        return {
            "kernel_name": kernel_name,
            "baseline": {
                "times": baseline_times,
                "latencies": baseline_history,
                "best_so_far": baseline_best,
            },
            "adaptive": {
                "times": adaptive_times,
                "latencies": adaptive_history,
                "best_so_far": adaptive_best,
            },
        }

    def save_plot_data(self, plot_data: Dict[str, Any], filename: str = "plot_data.json"):
        """Save plot data to JSON file."""
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(plot_data, indent=2))
        return output_path
