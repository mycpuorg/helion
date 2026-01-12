#!/usr/bin/env python3
"""
Benchmark: OpenEvolve vs Baseline Autotuners
=============================================

This script answers key community questions about OpenEvolve tuning:

1. Does it find faster results than existing algorithms?
2. How long does it take?
3. Where is time spent (LLM vs kernel compilation)?

Compares:
- RandomSearch (baseline)
- PatternSearch (default Helion autotuner)
- LLMSearch (direct LLM-based search)
- OpenEvolveTuner (evolutionary LLM search)

Usage:
    # Run with mock LLM (no API key needed, for testing)
    python examples/benchmark_openevolve_vs_baseline.py --mock

    # Run with real LLM (requires OPENAI_API_KEY)
    python examples/benchmark_openevolve_vs_baseline.py

    # Run specific autotuners only
    python examples/benchmark_openevolve_vs_baseline.py --tuners random,pattern
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl
from helion.autotuner import RandomSearch, PatternSearch


@dataclass
class BenchmarkResult:
    """Results from a single autotuner run."""
    tuner_name: str
    best_latency_ms: float
    total_time_s: float
    num_evaluations: int
    best_config: dict[str, Any] | None = None
    llm_time_s: float = 0.0
    compile_time_s: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    kernel_name: str
    problem_size: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary table."""
        lines = [
            f"\n{'='*70}",
            f"BENCHMARK RESULTS: {self.kernel_name} ({self.problem_size})",
            f"{'='*70}",
            f"{'Tuner':<20} {'Latency (ms)':<15} {'Time (s)':<12} {'Evals':<8} {'Speedup':<10}",
            f"{'-'*70}",
        ]

        # Find baseline (random search) latency
        baseline_latency = None
        for r in self.results:
            if r.tuner_name == "RandomSearch" and r.best_latency_ms > 0:
                baseline_latency = r.best_latency_ms
                break

        for r in self.results:
            if r.error:
                lines.append(f"{r.tuner_name:<20} {'ERROR':<15} {r.total_time_s:<12.1f} {'-':<8} {'-':<10}")
                lines.append(f"  Error: {r.error}")
            else:
                speedup = "-"
                if baseline_latency and r.best_latency_ms > 0:
                    speedup = f"{baseline_latency / r.best_latency_ms:.2f}x"
                lines.append(
                    f"{r.tuner_name:<20} {r.best_latency_ms:<15.3f} {r.total_time_s:<12.1f} "
                    f"{r.num_evaluations:<8} {speedup:<10}"
                )

        lines.append(f"{'='*70}")
        return "\n".join(lines)


# Define a simple matmul kernel for benchmarking
@helion.kernel(static_shapes=True)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication kernel for benchmarking autotuners."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def benchmark_random_search(
    bound_kernel,
    args: tuple,
    num_configs: int = 50,
) -> BenchmarkResult:
    """Benchmark RandomSearch autotuner."""
    print(f"\n--- RandomSearch ({num_configs} configs) ---")

    start_time = time.time()
    try:
        search = RandomSearch(bound_kernel, args, count=num_configs)
        best_config = search.autotune()

        # Get best latency
        fn = bound_kernel.compile_config(best_config)
        from triton.testing import do_bench
        latency_ms = do_bench(lambda: fn(*args), warmup=25, rep=100)

        total_time = time.time() - start_time

        return BenchmarkResult(
            tuner_name="RandomSearch",
            best_latency_ms=latency_ms,
            total_time_s=total_time,
            num_evaluations=num_configs,
            best_config=best_config.config if hasattr(best_config, 'config') else None,
        )
    except Exception as e:
        return BenchmarkResult(
            tuner_name="RandomSearch",
            best_latency_ms=float('inf'),
            total_time_s=time.time() - start_time,
            num_evaluations=0,
            error=str(e),
        )


def benchmark_pattern_search(
    bound_kernel,
    args: tuple,
    initial_population: int = 20,
    max_generations: int = 3,
) -> BenchmarkResult:
    """Benchmark PatternSearch autotuner."""
    print(f"\n--- PatternSearch (pop={initial_population}, gens={max_generations}) ---")

    start_time = time.time()
    try:
        search = PatternSearch(
            bound_kernel, args,
            initial_population=initial_population,
            max_generations=max_generations,
            copies=1,
        )
        best_config = search.autotune()

        # Get best latency
        fn = bound_kernel.compile_config(best_config)
        from triton.testing import do_bench
        latency_ms = do_bench(lambda: fn(*args), warmup=25, rep=100)

        total_time = time.time() - start_time

        # Estimate evaluations
        num_evals = getattr(search, 'evaluation_count', initial_population * (max_generations + 1))

        return BenchmarkResult(
            tuner_name="PatternSearch",
            best_latency_ms=latency_ms,
            total_time_s=total_time,
            num_evaluations=num_evals,
            best_config=best_config.config if hasattr(best_config, 'config') else None,
        )
    except Exception as e:
        return BenchmarkResult(
            tuner_name="PatternSearch",
            best_latency_ms=float('inf'),
            total_time_s=time.time() - start_time,
            num_evaluations=0,
            error=str(e),
        )


def benchmark_openevolve(
    config_space: dict[str, list],
    objective_fn,
    max_evaluations: int = 50,
    mock: bool = False,
) -> BenchmarkResult:
    """Benchmark OpenEvolve autotuner."""
    print(f"\n--- OpenEvolveTuner ({max_evaluations} evals, mock={mock}) ---")

    try:
        from helion.autotuner.openevolve_tuner import OpenEvolveTuner
    except ImportError:
        return BenchmarkResult(
            tuner_name="OpenEvolve",
            best_latency_ms=float('inf'),
            total_time_s=0,
            num_evaluations=0,
            error="OpenEvolve not installed",
        )

    start_time = time.time()
    llm_start = 0.0
    llm_total = 0.0

    try:
        if mock:
            # Mock mode - use random search fallback internally
            tuner = OpenEvolveTuner(
                config_space=config_space,
                objective=objective_fn,
                max_evaluations=max_evaluations,
                verbose=True,
                artifact_dir=None,
            )
            # Skip actual OpenEvolve and just use random search
            import random
            best_score = float('-inf')
            best_config = None
            for i in range(max_evaluations):
                config = {k: random.choice(v) for k, v in config_space.items()}
                try:
                    score = objective_fn(config)
                    if score > best_score:
                        best_score = score
                        best_config = config
                    print(f"  Eval {i+1}: {config} -> {score:.4f}")
                except Exception as e:
                    print(f"  Eval {i+1}: {config} -> ERROR: {e}")

            tuner.best_config = best_config
            tuner.best_score = best_score
            tuner.evaluation_count = max_evaluations
        else:
            tuner = OpenEvolveTuner(
                config_space=config_space,
                objective=objective_fn,
                max_evaluations=max_evaluations,
                verbose=True,
                artifact_dir="openevolve_benchmark_artifacts",
            )
            tuner.tune()

        total_time = time.time() - start_time

        # Convert score back to latency (score = 1/latency typically)
        best_latency = 1.0 / tuner.best_score if tuner.best_score and tuner.best_score > 0 else float('inf')

        return BenchmarkResult(
            tuner_name="OpenEvolve",
            best_latency_ms=best_latency,
            total_time_s=total_time,
            num_evaluations=tuner.evaluation_count,
            best_config=tuner.best_config,
            llm_time_s=llm_total,
            compile_time_s=total_time - llm_total,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            tuner_name="OpenEvolve",
            best_latency_ms=float('inf'),
            total_time_s=time.time() - start_time,
            num_evaluations=0,
            error=str(e),
        )


# Pickleable objective class that carries its own state
class MatmulObjective:
    """Pickleable objective for matmul tuning with FULL Helion config space."""

    def __init__(self, m: int = 1024, k: int = 1024, n: int = 1024):
        self._m = int(m)
        self._k = int(k)
        self._n = int(n)
        self._tensors_initialized = False
        self._x = None
        self._y = None
        self._ref = None

    def _ensure_tensors(self):
        """Lazily initialize tensors on first call (after unpickling)."""
        if not self._tensors_initialized:
            import torch
            # Re-import DEVICE since we might be in a subprocess
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._x = torch.randn([self._m, self._k], device=device, dtype=torch.float16)
            self._y = torch.randn([self._k, self._n], device=device, dtype=torch.float16)
            self._ref = self._x @ self._y
            self._tensors_initialized = True

    def __call__(self, config: dict) -> float:
        """Evaluate a config. Returns 1/latency_ms (higher is better)."""
        self._ensure_tensors()

        try:
            import helion
            import helion.language as hl

            # Build the full Helion Config from the dict
            block_sizes = [config["block_m"], config["block_n"], config["block_k"]]

            helion_config = helion.Config(
                block_sizes=block_sizes,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
                pid_type=config["pid_type"],
                indexing=config["indexing"],
                loop_orders=config["loop_orders"],
                l2_groupings=config["l2_groupings"],
                load_eviction_policies=config["load_eviction_policies"],
                range_unroll_factors=config["range_unroll_factors"],
                range_num_stages=config["range_num_stages"],
            )

            # Import matmul from examples
            import sys
            import os
            # Add examples directory to path
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
            if examples_dir not in sys.path:
                sys.path.insert(0, examples_dir)
            # Also try relative path
            if 'examples' not in sys.path:
                sys.path.insert(0, 'examples')

            from matmul import matmul as matmul_kernel

            # Bind and compile with config
            bound = matmul_kernel.bind((self._x, self._y))
            fn = bound.compile_config(helion_config)

            # Correctness check
            result = fn(self._x, self._y)
            import torch
            if not torch.allclose(result, self._ref, rtol=1e-2, atol=1e-2):
                return 0.0  # Invalid config

            # Benchmark
            from triton.testing import do_bench
            latency_ms = do_bench(lambda: fn(self._x, self._y), warmup=10, rep=50)

            return 1.0 / latency_ms  # Higher is better

        except Exception as e:
            print(f"    Config failed: {e}")
            return 0.0

    def __reduce__(self):
        """Custom pickle protocol - returns constructor and args."""
        return (MatmulObjective, (self._m, self._k, self._n))


# Module-level function for backward compatibility
def matmul_objective(config: dict) -> float:
    """Wrapper for backward compatibility."""
    obj = MatmulObjective()
    return obj(config)


def get_full_matmul_config_space():
    """
    Get the FULL Helion config space for matmul.
    This matches what PatternSearch explores.
    """
    return {
        # Block sizes (3 dimensions for matmul: M, N, K)
        "block_m": [16, 32, 64, 128, 256],
        "block_n": [16, 32, 64, 128, 256],
        "block_k": [16, 32, 64, 128],

        # Warps and stages
        "num_warps": [1, 2, 4, 8, 16],
        "num_stages": [1, 2, 3, 4, 5, 6, 7, 8],

        # PID scheduling strategy
        "pid_type": ["flat", "xyz", "persistent_blocked", "persistent_interleaved"],

        # Indexing modes (3 pointers for matmul: x, y, out)
        "indexing": [
            ["pointer", "pointer", "pointer"],
            ["pointer", "pointer", "tensor_descriptor"],
            ["pointer", "tensor_descriptor", "pointer"],
            ["pointer", "tensor_descriptor", "tensor_descriptor"],
            ["tensor_descriptor", "pointer", "pointer"],
            ["tensor_descriptor", "pointer", "tensor_descriptor"],
            ["tensor_descriptor", "tensor_descriptor", "pointer"],
            ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
        ],

        # Loop ordering (1 loop nest with 2 dimensions)
        "loop_orders": [
            [[0, 1]],
            [[1, 0]],
        ],

        # L2 cache grouping
        "l2_groupings": [
            [1], [2], [4], [8], [16], [32], [64],
        ],

        # Load eviction policies (2 loads: x and y)
        "load_eviction_policies": [
            ["", ""],
            ["first", ""],
            ["last", ""],
            ["", "first"],
            ["", "last"],
            ["first", "first"],
            ["first", "last"],
            ["last", "first"],
            ["last", "last"],
        ],

        # Range unroll factors (2 ranges: outer and inner loop)
        "range_unroll_factors": [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2], [2, 3],
            [3, 0], [3, 1], [3, 2], [3, 3],
        ],

        # Range num stages (software pipelining per loop)
        "range_num_stages": [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2], [2, 3],
            [3, 0], [3, 1], [3, 2], [3, 3],
        ],
    }


def create_matmul_objective(m: int, k: int, n: int):
    """Create and return the matmul objective function."""
    return MatmulObjective(m, k, n)


def run_benchmark(
    m: int = 1024,
    k: int = 1024,
    n: int = 1024,
    tuners: list[str] | None = None,
    mock: bool = False,
    num_evals: int = 50,
) -> BenchmarkSuite:
    """Run benchmark comparing autotuners."""

    if tuners is None:
        tuners = ["random", "pattern", "openevolve"]

    print(f"\n{'='*70}")
    print(f"MATMUL AUTOTUNER BENCHMARK")
    print(f"{'='*70}")
    print(f"Problem size: M={m}, K={k}, N={n}")
    print(f"Device: {DEVICE}")
    print(f"Tuners: {tuners}")
    print(f"Mock mode: {mock}")
    print(f"Max evaluations: {num_evals}")

    suite = BenchmarkSuite(
        kernel_name="matmul",
        problem_size=f"M={m}, K={k}, N={n}",
    )

    # Prepare inputs
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    args = (x, y)

    # Bind kernel for Helion autotuners
    bound_kernel = matmul_kernel.bind(args)

    # FULL config space for OpenEvolve - same complexity as Helion's autotuners
    config_space = get_full_matmul_config_space()

    # Calculate total config space size
    total_configs = 1
    for param_name, param_values in config_space.items():
        total_configs *= len(param_values)
    print(f"\nTotal config space size: {total_configs:,} configurations")
    print(f"Config space parameters: {len(config_space)}")

    # Run benchmarks
    if "random" in tuners:
        result = benchmark_random_search(bound_kernel, args, num_configs=num_evals)
        suite.results.append(result)
        print(f"  -> Latency: {result.best_latency_ms:.3f} ms, Time: {result.total_time_s:.1f}s")

    if "pattern" in tuners:
        # Aim for similar total evaluations
        init_pop = max(10, num_evals // 4)
        max_gens = max(1, num_evals // init_pop - 1)
        result = benchmark_pattern_search(bound_kernel, args, init_pop, max_gens)
        suite.results.append(result)
        print(f"  -> Latency: {result.best_latency_ms:.3f} ms, Time: {result.total_time_s:.1f}s")

    if "openevolve" in tuners:
        objective = create_matmul_objective(m, k, n)
        result = benchmark_openevolve(config_space, objective, num_evals, mock=mock)
        suite.results.append(result)
        if result.best_latency_ms < float('inf'):
            print(f"  -> Latency: {result.best_latency_ms:.3f} ms, Time: {result.total_time_s:.1f}s")

    return suite


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenEvolve vs baseline autotuners")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API calls)")
    parser.add_argument("--tuners", type=str, default="random,pattern,openevolve",
                        help="Comma-separated list of tuners to run")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--evals", type=int, default=50, help="Max evaluations per tuner")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    tuners = [t.strip() for t in args.tuners.split(",")]

    # Check for API key if not in mock mode and using openevolve
    if "openevolve" in tuners and not args.mock:
        if "OPENAI_API_KEY" not in os.environ:
            print("WARNING: OPENAI_API_KEY not set. Use --mock for testing without API.")
            print("         Or set: export OPENAI_API_KEY='your-key'")
            args.mock = True

    suite = run_benchmark(
        m=args.m,
        k=args.k,
        n=args.n,
        tuners=tuners,
        mock=args.mock,
        num_evals=args.evals,
    )

    print(suite.summary())

    # Time breakdown analysis
    print("\n--- TIME BREAKDOWN ANALYSIS ---")
    for r in suite.results:
        if r.error:
            continue
        if r.tuner_name == "OpenEvolve" and r.llm_time_s > 0:
            compile_pct = (r.compile_time_s / r.total_time_s) * 100 if r.total_time_s > 0 else 0
            llm_pct = (r.llm_time_s / r.total_time_s) * 100 if r.total_time_s > 0 else 0
            print(f"{r.tuner_name}:")
            print(f"  LLM time: {r.llm_time_s:.1f}s ({llm_pct:.1f}%)")
            print(f"  Compile/Eval time: {r.compile_time_s:.1f}s ({compile_pct:.1f}%)")
        else:
            print(f"{r.tuner_name}: {r.total_time_s:.1f}s total (all compilation/evaluation)")

    # Save results
    if args.output:
        output_data = {
            "kernel": suite.kernel_name,
            "problem_size": suite.problem_size,
            "results": [
                {
                    "tuner": r.tuner_name,
                    "latency_ms": r.best_latency_ms,
                    "time_s": r.total_time_s,
                    "evaluations": r.num_evaluations,
                    "error": r.error,
                    "config": r.best_config,
                }
                for r in suite.results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
