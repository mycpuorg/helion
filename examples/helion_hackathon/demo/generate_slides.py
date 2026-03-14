"""
Demo Slide Generator.

Creates presentation materials from pipeline results.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SlideContent:
    """Content for a single slide."""
    title: str
    content: str
    notes: Optional[str] = None


class SlideGenerator:
    """
    Generates demo presentation materials.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_title_slide(self) -> SlideContent:
        """Generate title slide."""
        return SlideContent(
            title="EvoX-Helion + Magpie",
            content="""
# EvoX-Helion + Magpie
## Adaptive Autotuning for AMD MI350X

**The Story in One Sentence:**

*"We used Magpie to find which kernels matter most in real AMD workloads,
applied EvoX-style adaptive autotuning to Helion on those kernels,
and showed side-by-side latency speedups on MI350X."*

---

**Why This is Different:**
- AMD hardware focus (MI350X)
- AMD tooling integration (Helion)
- Research paper insights applied to open questions
""",
            notes="This differentiates us from every other team at the hackathon"
        )

    def generate_kernel_selection_slide(
        self,
        selection_results: List[Dict[str, Any]],
    ) -> SlideContent:
        """Generate Slide 1: Kernel Selection."""

        # Build table
        table_rows = []
        total_wall_time = 0

        for r in selection_results:
            name = r.get('kernel_name', 'Unknown')
            wall_time = r.get('wall_time_fraction', 0) * 100
            config_space = r.get('config_space_size', 0)
            score = r.get('selection_score', 0)
            total_wall_time += r.get('wall_time_fraction', 0)

            space_str = f"{config_space:.1e}" if config_space > 1000 else str(config_space)
            table_rows.append(f"| {name} | {wall_time:.1f}% | {space_str} | {score:.3f} |")

        table = "\n".join([
            "| Kernel Name | Wall-time % | Config Space | Selection Score |",
            "|-------------|-------------|--------------|-----------------|",
        ] + table_rows)

        content = f"""
# Slide 1: Kernel Selection (Magpie)

## Target Kernels for Autotuning

{table}

---

**Selection Formula:** `score = wall_time_fraction × log(config_space_size)`

**Why These Kernels?**
- These {len(selection_results)} kernels represent **{total_wall_time*100:.1f}%** of model execution time
- Large config spaces provide room for adaptive autotuning to find better configs
- High wall-time kernels give maximum latency improvement per optimization effort

*"We didn't just pick LayerNorm because it's famous — Magpie told us it accounts for
significant wall time and has large configuration space."*
"""
        return SlideContent(
            title="Kernel Selection",
            content=content,
            notes="Explain the selection formula and why it prioritizes both impact and opportunity"
        )

    def generate_autotuning_slide(
        self,
        comparison_results: List[Dict[str, Any]],
    ) -> SlideContent:
        """Generate Slide 2: Adaptive Autotuning Results."""

        # Latency table
        latency_rows = []
        for r in comparison_results:
            name = r.get('kernel_name', 'Unknown')
            baseline = r.get('baseline_latency_ms', 0)
            adaptive = r.get('adaptive_latency_ms', 0)
            speedup = r.get('speedup', 1)
            latency_rows.append(f"| {name} | {baseline:.4f} | {adaptive:.4f} | **{speedup:.2f}x** |")

        latency_table = "\n".join([
            "| Kernel | Baseline (ms) | Adaptive (ms) | Speedup |",
            "|--------|---------------|---------------|---------|",
        ] + latency_rows)

        # Time table
        time_rows = []
        for r in comparison_results:
            name = r.get('kernel_name', 'Unknown')
            base_time = r.get('baseline_time_to_best_s', 0)
            adapt_time = r.get('adaptive_time_to_best_s', 0)
            time_speedup = r.get('time_speedup', 1)
            time_rows.append(f"| {name} | {base_time:.1f} | {adapt_time:.1f} | **{time_speedup:.2f}x** |")

        time_table = "\n".join([
            "| Kernel | Baseline Time (s) | Adaptive Time (s) | Speedup |",
            "|--------|-------------------|-------------------|---------|",
        ] + time_rows)

        content = f"""
# Slide 2: Adaptive Autotuning (EvoX + Helion)

## Latency Improvements

{latency_table}

---

## Time-to-Best Performance

{time_table}

---

**Key Insight:** The adaptive strategy not only finds better configs,
but finds them **faster** with **fewer evaluations**.

**EvoX Principle:** The selection mechanism itself should adapt based on
search progress — not just use fixed strategies.
"""
        return SlideContent(
            title="Autotuning Results",
            content=content,
            notes="Emphasize both the final latency improvement AND the faster convergence"
        )

    def generate_comparison_slide(
        self,
        report: Dict[str, Any],
    ) -> SlideContent:
        """Generate Slide 3: Side-by-Side Results."""

        summary = report.get('summary', {})
        kernels = report.get('kernels', [])

        # Final comparison table
        rows = []
        for k in kernels:
            name = k.get('kernel_name', 'Unknown')
            latency = k.get('adaptive_latency_ms', 0)
            baseline = k.get('baseline_latency_ms', 0)
            improvement = ((baseline - latency) / baseline * 100) if baseline > 0 else 0
            efficiency = k.get('eval_efficiency', 1)
            rows.append(f"| {name} | {latency:.4f} | {improvement:.1f}% | {efficiency:.2f}x |")

        table = "\n".join([
            "| Kernel | Best Latency (ms) | Improvement | Eval Efficiency |",
            "|--------|-------------------|-------------|-----------------|",
        ] + rows)

        content = f"""
# Slide 3: Side-by-Side Results (Magpie Compare)

**Hardware:** {report.get('hardware', 'AMD MI350X')}

## Final Comparison

{table}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average Latency Speedup | **{summary.get('avg_latency_speedup', 1):.2f}x** |
| Average Time-to-Best Speedup | **{summary.get('avg_time_to_best_speedup', 1):.2f}x** |
| Average Evaluation Efficiency | **{summary.get('avg_eval_efficiency', 1):.2f}x** |

---

**The EvoX Efficiency Argument:**
- Not just better final quality
- But better quality **faster**
- With **fewer evaluations**
"""
        return SlideContent(
            title="Final Results",
            content=content,
            notes="This is the money slide - emphasize all three metrics together"
        )

    def generate_full_presentation(
        self,
        selection_results: List[Dict[str, Any]],
        comparison_results: List[Dict[str, Any]],
        full_report: Dict[str, Any],
    ) -> str:
        """
        Generate complete presentation markdown.

        Returns:
            Full markdown content for the presentation
        """
        slides = [
            self.generate_title_slide(),
            self.generate_kernel_selection_slide(selection_results),
            self.generate_autotuning_slide(comparison_results),
            self.generate_comparison_slide(full_report),
        ]

        content = []
        for i, slide in enumerate(slides):
            content.append(slide.content)
            if i < len(slides) - 1:
                content.append("\n---\n")

        return "\n".join(content)

    def save_presentation(
        self,
        selection_results: List[Dict[str, Any]],
        comparison_results: List[Dict[str, Any]],
        full_report: Dict[str, Any],
        filename: str = "presentation.md",
    ) -> Path:
        """
        Generate and save the presentation.

        Returns:
            Path to the saved file
        """
        content = self.generate_full_presentation(
            selection_results,
            comparison_results,
            full_report,
        )
        output_path = self.output_dir / filename
        output_path.write_text(content)
        return output_path
