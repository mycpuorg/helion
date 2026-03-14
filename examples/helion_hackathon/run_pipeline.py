#!/usr/bin/env python3
"""
EvoX-Helion + Magpie Pipeline Orchestration.

This is the main entry point for running the complete hackathon pipeline:
1. Stage 1: Kernel profiling and selection (Magpie-style)
2. Stage 2: Baseline and adaptive autotuning
3. Stage 3: Comparison and demo generation

Usage:
    python run_pipeline.py --mode full        # Run everything
    python run_pipeline.py --mode profile     # Stage 1 only
    python run_pipeline.py --mode autotune    # Stage 2 only
    python run_pipeline.py --mode compare     # Stage 3 only
    python run_pipeline.py --mode demo        # Generate demo slides
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_stage1_profiling(
    device: str = "cuda",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Profile kernels and select targets for autotuning.

    Returns:
        List of kernel selection results
    """
    from stage1_magpie.kernel_profiler import KernelProfiler
    from stage1_magpie.kernel_selector import KernelSelector, rank_kernels
    from kernels.layernorm import LayerNormConfig
    from kernels.softmax import SoftmaxConfig
    from kernels.matmul import MatmulConfig
    from kernels.attention import AttentionConfig
    from kernels.rope import RoPEConfig

    if verbose:
        print("=" * 70)
        print("STAGE 1: Kernel Profiling and Selection")
        print("=" * 70)

    # Define kernel configurations
    kernel_configs = {
        "layernorm": LayerNormConfig().config_space_size,
        "softmax": SoftmaxConfig().config_space_size,
        "matmul": MatmulConfig().config_space_size,
        "attention": AttentionConfig().config_space_size,
        "rope": RoPEConfig().config_space_size,
    }

    if verbose:
        print("\nKernel Configuration Spaces:")
        for name, size in kernel_configs.items():
            print(f"  {name}: {size:,} configurations")

    # Create profiler
    profiler = KernelProfiler(warmup_iterations=5, profile_iterations=50)
    for name, size in kernel_configs.items():
        profiler.register_kernel(name, size)

    # Simulate profiling with realistic wall-time fractions
    # (In production, this would use actual CUDA profiling)
    from stage1_magpie.kernel_profiler import ProfileResult

    simulated_results = [
        ProfileResult(
            kernel_name="matmul",
            total_time_ms=450.0,
            call_count=200,
            avg_time_ms=2.25,
            min_time_ms=2.1,
            max_time_ms=2.5,
            config_space_size=kernel_configs["matmul"],
            wall_time_fraction=0.35,
        ),
        ProfileResult(
            kernel_name="attention",
            total_time_ms=320.0,
            call_count=50,
            avg_time_ms=6.4,
            min_time_ms=6.0,
            max_time_ms=7.0,
            config_space_size=kernel_configs["attention"],
            wall_time_fraction=0.25,
        ),
        ProfileResult(
            kernel_name="layernorm",
            total_time_ms=230.0,
            call_count=100,
            avg_time_ms=2.3,
            min_time_ms=2.1,
            max_time_ms=2.6,
            config_space_size=kernel_configs["layernorm"],
            wall_time_fraction=0.18,
        ),
        ProfileResult(
            kernel_name="softmax",
            total_time_ms=150.0,
            call_count=50,
            avg_time_ms=3.0,
            min_time_ms=2.8,
            max_time_ms=3.3,
            config_space_size=kernel_configs["softmax"],
            wall_time_fraction=0.12,
        ),
        ProfileResult(
            kernel_name="rope",
            total_time_ms=130.0,
            call_count=50,
            avg_time_ms=2.6,
            min_time_ms=2.4,
            max_time_ms=2.9,
            config_space_size=kernel_configs["rope"],
            wall_time_fraction=0.10,
        ),
    ]

    if verbose:
        print("\n" + profiler.format_profile_table(simulated_results))

    # Select top kernels
    selection_results, explanation = rank_kernels(simulated_results, top_k=5)

    if verbose:
        print("\n" + explanation)

    # Convert to dicts for JSON serialization
    results = [r.to_dict() for r in selection_results]

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "stage1_selection.json").write_text(json.dumps(results, indent=2))

    return results


def run_stage2_autotuning(
    kernel_name: str = "layernorm",
    max_evaluations: int = 200,
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """
    Stage 2: Run baseline and adaptive autotuning.

    Returns:
        Tuple of (baseline_result, adaptive_result)
    """
    from stage2_evox_autotuning.lfbo_wrapper import (
        AdaptiveLFBOPatternSearch,
        BaselineLFBOPatternSearch,
    )
    from kernels.layernorm import LayerNormConfig, get_layernorm_config_space
    from kernels.softmax import SoftmaxConfig, get_softmax_config_space
    from kernels.matmul import MatmulConfig, get_matmul_config_space
    from kernels.attention import AttentionConfig, get_attention_config_space
    from kernels.rope import RoPEConfig, get_rope_config_space
    import random

    if verbose:
        print("=" * 70)
        print(f"STAGE 2: Autotuning - {kernel_name}")
        print("=" * 70)

    # Get config space for the kernel
    config_spaces = {
        "layernorm": (get_layernorm_config_space, LayerNormConfig),
        "softmax": (get_softmax_config_space, SoftmaxConfig),
        "matmul": (get_matmul_config_space, MatmulConfig),
        "attention": (get_attention_config_space, AttentionConfig),
        "rope": (get_rope_config_space, RoPEConfig),
    }

    get_space_fn, config_cls = config_spaces.get(kernel_name, config_spaces["layernorm"])
    configs = get_space_fn()

    # Limit config space for demo
    if len(configs) > 500:
        configs = random.sample(configs, 500)

    # Convert to dicts
    config_dicts = [c.to_dict() for c in configs]

    if verbose:
        print(f"Config space size: {len(config_dicts)}")

    # Create benchmark function (simulated for demo)
    # In production, this would compile and benchmark actual kernels
    def benchmark_fn(config: Dict[str, Any]) -> Tuple[float, bool]:
        """Simulate kernel benchmarking with realistic latency landscape."""
        import random

        # Create a synthetic but realistic latency landscape
        base_latency = 2.0

        # Block size effect
        block_size = config.get("BLOCK_SIZE", config.get("BLOCK_SIZE_M", 1024))
        if block_size == 512:
            base_latency *= 0.95
        elif block_size == 1024:
            base_latency *= 1.0
        elif block_size == 2048:
            base_latency *= 1.1

        # Warps effect
        num_warps = config.get("num_warps", 4)
        if num_warps == 4:
            base_latency *= 0.98
        elif num_warps == 8:
            base_latency *= 1.0
        elif num_warps == 16:
            base_latency *= 1.05

        # Stages effect
        num_stages = config.get("num_stages", 2)
        if num_stages == 2:
            base_latency *= 0.97
        elif num_stages == 3:
            base_latency *= 1.0
        elif num_stages == 4:
            base_latency *= 1.02

        # Add noise
        noise = random.gauss(0, 0.05)
        latency = base_latency * (1 + noise)

        # Simulate occasional errors
        is_error = random.random() < 0.02

        return (latency, is_error)

    # Create encoding function
    def encode_fn(config: Dict[str, Any]) -> Tuple[int, ...]:
        """Encode config for diversity computation."""
        values = []
        for key in sorted(config.keys()):
            val = config[key]
            if isinstance(val, (int, float)):
                values.append(int(val) % 100)  # Simple encoding
        return tuple(values)

    # Run baseline
    if verbose:
        print("\n--- Running Baseline LFBO ---")

    baseline_search = BaselineLFBOPatternSearch(
        kernel_fn=None,  # Not used in simulation
        config_space=config_dicts,
        benchmark_fn=benchmark_fn,
        encode_fn=encode_fn,
    )
    baseline_result = baseline_search.autotune(
        max_evaluations=max_evaluations,
        initial_population_size=30,
        batch_size=15,
        verbose=verbose,
    )

    # Run adaptive
    if verbose:
        print("\n--- Running Adaptive EvoX ---")

    adaptive_search = AdaptiveLFBOPatternSearch(
        kernel_fn=None,
        config_space=config_dicts,
        benchmark_fn=benchmark_fn,
        encode_fn=encode_fn,
    )
    adaptive_result = adaptive_search.autotune(
        max_evaluations=max_evaluations,
        initial_population_size=30,
        batch_size=15,
        verbose=verbose,
    )

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    baseline_data = {
        "kernel_name": kernel_name,
        "best_latency": baseline_result.best_latency,
        "total_evaluations": baseline_result.total_evaluations,
        "total_time": baseline_result.total_time_seconds,
        "latency_history": baseline_result.latency_history,
        "time_history": baseline_result.time_history,
    }

    adaptive_data = {
        "kernel_name": kernel_name,
        "best_latency": adaptive_result.best_latency,
        "total_evaluations": adaptive_result.total_evaluations,
        "total_time": adaptive_result.total_time_seconds,
        "latency_history": adaptive_result.latency_history,
        "time_history": adaptive_result.time_history,
        "controller_history": adaptive_result.controller_history,
    }

    (output_dir / f"stage2_baseline_{kernel_name}.json").write_text(
        json.dumps(baseline_data, indent=2)
    )
    (output_dir / f"stage2_adaptive_{kernel_name}.json").write_text(
        json.dumps(adaptive_data, indent=2)
    )

    return baseline_result, adaptive_result


def run_stage3_comparison(
    results: Dict[str, Tuple[Any, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stage 3: Compare baseline vs adaptive results.

    Args:
        results: Dict of kernel_name -> (baseline_result, adaptive_result)

    Returns:
        Full comparison report
    """
    from stage3_comparison.magpie_compare import KernelComparator

    if verbose:
        print("=" * 70)
        print("STAGE 3: Comparison and Evaluation")
        print("=" * 70)

    comparator = KernelComparator(hardware="AMD MI350X", output_dir=PROJECT_ROOT / "results")

    for kernel_name, (baseline, adaptive) in results.items():
        comparator.compare_kernel(kernel_name, baseline, adaptive)

    report = comparator.generate_report()

    if verbose:
        print("\n" + comparator.format_comparison_table())

    # Save report
    output_path = comparator.save_report("stage3_comparison.json")
    if verbose:
        print(f"\nReport saved to: {output_path}")

    return {
        "timestamp": report.timestamp,
        "hardware": report.hardware,
        "kernels": [asdict(k) for k in report.kernels],
        "summary": report.summary,
        "testcases": report.testcases,
        "metrics": report.metrics,
    }


def generate_demo(
    selection_results: List[Dict],
    comparison_report: Dict,
    verbose: bool = True,
):
    """Generate demo slides and plots."""
    from demo.generate_slides import SlideGenerator
    from demo.plot_latency_curves import LatencyCurvePlotter

    if verbose:
        print("=" * 70)
        print("GENERATING DEMO MATERIALS")
        print("=" * 70)

    output_dir = PROJECT_ROOT / "demo"
    output_dir.mkdir(exist_ok=True)

    # Generate slides
    generator = SlideGenerator(output_dir=output_dir)
    comparison_results = comparison_report.get("kernels", [])

    slides_path = generator.save_presentation(
        selection_results=selection_results,
        comparison_results=comparison_results,
        full_report=comparison_report,
    )

    if verbose:
        print(f"\nPresentation saved to: {slides_path}")

    # Generate plots
    plotter = LatencyCurvePlotter(output_dir=output_dir / "plots")

    # Generate speedup chart
    kernel_names = [k["kernel_name"] for k in comparison_results]
    speedups = [k["speedup"] for k in comparison_results]
    time_speedups = [k["time_speedup"] for k in comparison_results]

    speedup_path = plotter.plot_speedup_bar_chart(kernel_names, speedups, time_speedups)
    if verbose:
        print(f"Speedup chart saved to: {speedup_path}")


def run_full_pipeline(
    kernels: Optional[List[str]] = None,
    max_evaluations: int = 200,
    verbose: bool = True,
):
    """Run the complete pipeline end-to-end."""
    if kernels is None:
        kernels = ["layernorm", "softmax", "matmul"]

    start_time = time.time()

    print("\n" + "=" * 70)
    print("EVOX-HELION + MAGPIE PIPELINE")
    print("Adaptive Autotuning for AMD MI350X")
    print("=" * 70 + "\n")

    # Stage 1: Profiling and Selection
    print("\n[1/4] Running Stage 1: Kernel Profiling...")
    selection_results = run_stage1_profiling(verbose=verbose)

    # Stage 2: Autotuning for each kernel
    print("\n[2/4] Running Stage 2: Autotuning...")
    autotune_results = {}
    for kernel in kernels:
        baseline, adaptive = run_stage2_autotuning(
            kernel_name=kernel,
            max_evaluations=max_evaluations,
            verbose=verbose,
        )
        autotune_results[kernel] = (baseline, adaptive)

    # Stage 3: Comparison
    print("\n[3/4] Running Stage 3: Comparison...")
    comparison_report = run_stage3_comparison(autotune_results, verbose=verbose)

    # Demo Generation
    print("\n[4/4] Generating Demo Materials...")
    generate_demo(selection_results, comparison_report, verbose=verbose)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_time:.1f} seconds")
    print("=" * 70)
    print("\nOutput files:")
    print("  - results/stage1_selection.json")
    print("  - results/stage2_baseline_*.json")
    print("  - results/stage2_adaptive_*.json")
    print("  - results/stage3_comparison.json")
    print("  - demo/presentation.md")
    print("  - demo/plots/speedup_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="EvoX-Helion + Magpie Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "profile", "autotune", "compare", "demo"],
        default="full",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["layernorm", "softmax", "matmul"],
        help="Kernels to autotune",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=200,
        help="Maximum evaluations per kernel",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "full":
        run_full_pipeline(
            kernels=args.kernels,
            max_evaluations=args.max_evals,
            verbose=verbose,
        )
    elif args.mode == "profile":
        run_stage1_profiling(verbose=verbose)
    elif args.mode == "autotune":
        for kernel in args.kernels:
            run_stage2_autotuning(
                kernel_name=kernel,
                max_evaluations=args.max_evals,
                verbose=verbose,
            )
    elif args.mode == "compare":
        # Load existing results and compare
        print("Loading existing results for comparison...")
        # This would load from saved JSON files
    elif args.mode == "demo":
        # Load results and generate demo
        results_dir = PROJECT_ROOT / "results"
        if (results_dir / "stage1_selection.json").exists():
            selection = json.loads((results_dir / "stage1_selection.json").read_text())
            comparison = json.loads((results_dir / "stage3_comparison.json").read_text())
            generate_demo(selection, comparison, verbose=verbose)
        else:
            print("No results found. Run full pipeline first.")


if __name__ == "__main__":
    main()
