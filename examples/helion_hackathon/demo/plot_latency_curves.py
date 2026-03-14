"""
Latency Curve Plotter for Demo Visualization.

Generates latency-vs-time plots comparing baseline and adaptive autotuning,
similar to Figure 4 in the LFBO blog post.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class LatencyCurvePlotter:
    """
    Generates latency-vs-time curve plots for autotuning comparison.

    Creates visualizations showing:
    1. Latency of best config over time
    2. Comparison between baseline and adaptive strategies
    3. Convergence characteristics
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("demo/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_best_so_far(self, latencies: List[float]) -> List[float]:
        """Compute running minimum (best latency so far)."""
        best_so_far = []
        current_best = float("inf")

        for lat in latencies:
            if lat < current_best and lat != float("inf"):
                current_best = lat
            best_so_far.append(current_best if current_best != float("inf") else lat)

        return best_so_far

    def plot_comparison(
        self,
        kernel_name: str,
        baseline_latencies: List[float],
        baseline_times: List[float],
        adaptive_latencies: List[float],
        adaptive_times: List[float],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Create a comparison plot for a single kernel.

        Args:
            kernel_name: Name of the kernel
            baseline_latencies: Latency history from baseline search
            baseline_times: Time history from baseline search
            adaptive_latencies: Latency history from adaptive search
            adaptive_times: Time history from adaptive search
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            # Fallback: save data as JSON for external plotting
            return self._save_plot_data(
                kernel_name,
                baseline_latencies, baseline_times,
                adaptive_latencies, adaptive_times,
            )

        # Compute best-so-far curves
        baseline_best = self.compute_best_so_far(baseline_latencies)
        adaptive_best = self.compute_best_so_far(adaptive_latencies)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Latency over time
        ax1.plot(baseline_times, baseline_best, 'b-', linewidth=2, label='Baseline LFBO')
        ax1.plot(adaptive_times, adaptive_best, 'r-', linewidth=2, label='Adaptive EvoX')

        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Best Latency (ms)', fontsize=12)
        ax1.set_title(f'{kernel_name}: Latency vs Time', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Latency over evaluations
        ax2.plot(range(len(baseline_best)), baseline_best, 'b-', linewidth=2, label='Baseline LFBO')
        ax2.plot(range(len(adaptive_best)), adaptive_best, 'r-', linewidth=2, label='Adaptive EvoX')

        ax2.set_xlabel('Number of Evaluations', fontsize=12)
        ax2.set_ylabel('Best Latency (ms)', fontsize=12)
        ax2.set_title(f'{kernel_name}: Latency vs Evaluations', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"{kernel_name}_comparison.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_all_kernels(
        self,
        results: List[Dict[str, Any]],
    ) -> Path:
        """
        Create a grid plot comparing all kernels.

        Args:
            results: List of result dicts with latency/time histories

        Returns:
            Path to saved plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            # Fallback to JSON
            json_path = self.output_dir / "all_kernels_data.json"
            json_path.write_text(json.dumps(results, indent=2))
            return json_path

        n_kernels = len(results)
        if n_kernels == 0:
            return self.output_dir / "empty.png"

        # Create subplot grid
        cols = 2
        rows = (n_kernels + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            kernel_name = result.get('kernel_name', f'Kernel {idx}')

            # Get data
            baseline_latencies = result.get('baseline_latencies', [])
            baseline_times = result.get('baseline_times', [])
            adaptive_latencies = result.get('adaptive_latencies', [])
            adaptive_times = result.get('adaptive_times', [])

            # Compute best-so-far
            baseline_best = self.compute_best_so_far(baseline_latencies)
            adaptive_best = self.compute_best_so_far(adaptive_latencies)

            # Plot
            if baseline_times and baseline_best:
                ax.plot(baseline_times, baseline_best, 'b-', linewidth=2, label='Baseline')
            if adaptive_times and adaptive_best:
                ax.plot(adaptive_times, adaptive_best, 'r-', linewidth=2, label='Adaptive')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(kernel_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_kernels, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)

        plt.suptitle('Latency Convergence: Baseline vs Adaptive', fontsize=16)
        plt.tight_layout()

        save_path = self.output_dir / "all_kernels_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_speedup_bar_chart(
        self,
        kernel_names: List[str],
        speedups: List[float],
        time_speedups: List[float],
    ) -> Path:
        """
        Create a bar chart showing speedups across kernels.

        Args:
            kernel_names: Names of kernels
            speedups: Latency speedup factors
            time_speedups: Time-to-best speedup factors

        Returns:
            Path to saved plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
        except ImportError:
            json_path = self.output_dir / "speedup_data.json"
            json_path.write_text(json.dumps({
                "kernels": kernel_names,
                "speedups": speedups,
                "time_speedups": time_speedups,
            }, indent=2))
            return json_path

        x = np.arange(len(kernel_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width/2, speedups, width, label='Latency Speedup', color='steelblue')
        bars2 = ax.bar(x + width/2, time_speedups, width, label='Time-to-Best Speedup', color='coral')

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax.set_xlabel('Kernel', fontsize=12)
        ax.set_title('Adaptive vs Baseline Speedups by Kernel', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_names, rotation=45, ha='right')
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        save_path = self.output_dir / "speedup_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def _save_plot_data(
        self,
        kernel_name: str,
        baseline_latencies: List[float],
        baseline_times: List[float],
        adaptive_latencies: List[float],
        adaptive_times: List[float],
    ) -> Path:
        """Fallback: save plot data as JSON when matplotlib unavailable."""
        data = {
            "kernel_name": kernel_name,
            "baseline": {
                "latencies": baseline_latencies,
                "times": baseline_times,
                "best_so_far": self.compute_best_so_far(baseline_latencies),
            },
            "adaptive": {
                "latencies": adaptive_latencies,
                "times": adaptive_times,
                "best_so_far": self.compute_best_so_far(adaptive_latencies),
            },
        }

        save_path = self.output_dir / f"{kernel_name}_plot_data.json"
        save_path.write_text(json.dumps(data, indent=2))
        return save_path
