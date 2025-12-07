"""
SLM Attention Kernel Tuning with OpenEvolve
============================================

This script uses the OpenEvolve autotuner to find optimal attention kernel
configurations for Small Language Models (SLMs) on NVIDIA B200.

The attention kernel is the 3D batched flash attention from transformer.py.

Requirements:
- NVIDIA B200 GPU
- OpenEvolve: pip install openevolve
- OPENAI_API_KEY environment variable (or run in mock mode)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE, get_nvidia_gpu_model

# Check for OpenEvolve
try:
    from helion.autotuner.openevolve_tuner import OpenEvolveTuner
    HAS_OPENEVOLVE = True
except ImportError:
    HAS_OPENEVOLVE = False
    print("Warning: OpenEvolve not installed. Install with: pip install openevolve")

# Check API key
MOCK_MODE = "OPENAI_API_KEY" not in os.environ


def create_attention_kernel_with_config(config: dict[str, Any]):
    """
    Create a 3D flash attention kernel with a specific helion config.

    The config dict should contain:
    - block_b: Block size for batch dimension
    - block_m: Block size for sequence (M) dimension
    - block_n: Block size for sequence (N) dimension
    - num_warps: Number of warps
    - num_stages: Pipeline stages
    """
    block_sizes = [config["block_b"], config["block_m"], config["block_n"]]

    @helion.kernel(
        static_shapes=True,
        config=helion.Config(
            block_sizes=block_sizes,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        ),
    )
    def attention(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
    ) -> torch.Tensor:
        """3D batched flash attention with online softmax."""
        m_dim = q_in.size(-2)
        n_dim = k_in.size(-2)
        assert n_dim == v_in.size(-2)
        head_dim = hl.specialize(q_in.size(-1))
        assert head_dim == k_in.size(-1) == v_in.size(-1)
        q_view = q_in.reshape([-1, m_dim, head_dim])
        v_view = v_in.reshape([-1, n_dim, head_dim])
        k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
        out = torch.empty_like(q_view)
        sm_scale = 1.0 / math.sqrt(head_dim)
        qk_scale = sm_scale * 1.44269504
        for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
            m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
            l_i = torch.full_like(m_i, 1.0)
            acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
            q = q_view[tile_b, tile_m, :]
            for tile_n in hl.tile(v_view.size(1)):
                k = k_view[tile_b, :, tile_n]
                qk = torch.bmm(q, k)
                m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, :, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, :, None]
                v = v_view[tile_b, tile_n, :]
                p = p.to(v.dtype)
                acc = torch.baddbmm(acc, p, v)
                m_i = m_ij
            m_i += torch.log2(l_i)
            acc = acc / l_i[:, :, None]
            out[tile_b, tile_m, :] = acc.to(out.dtype)
        return out.view(q_in.size())

    return attention


def evaluate_attention_config(config: dict[str, Any]) -> float:
    """
    Benchmark an attention configuration.

    Returns TFLOPS (higher is better), or 0.0 for failed configs.
    """
    try:
        # Create kernel with this config
        attention = create_attention_kernel_with_config(config)

        # Test problem size: typical SLM attention shape
        # Batch=1, Heads=16, SeqLen=1024, HeadDim=64
        batch_heads = 16
        seq_len = 1024
        head_dim = 64

        q = torch.randn(batch_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        k = torch.randn(batch_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        v = torch.randn(batch_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)

        # Correctness check
        y_helion = attention(q, k, v)
        y_ref = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)

        if not torch.allclose(y_helion, y_ref, rtol=1e-2, atol=1e-2):
            print(f"  Config {config}: FAILED correctness check")
            return 0.0

        # Benchmark
        from triton.testing import do_bench
        time_ms = do_bench(lambda: attention(q, k, v), warmup=25, rep=100)

        # Calculate TFLOPS
        # Attention: 2 * B * S^2 * D (for Q@K^T) + 2 * B * S^2 * D (for scores@V)
        flops = 4 * batch_heads * seq_len * seq_len * head_dim
        tflops = (flops / (time_ms * 1e-3)) / 1e12

        print(f"  Config block=[{config['block_b']},{config['block_m']},{config['block_n']}] "
              f"warps={config['num_warps']} stages={config['num_stages']}: "
              f"{time_ms:.3f}ms, {tflops:.2f} TFLOPS")

        return tflops

    except torch.cuda.OutOfMemoryError:
        print(f"  Config {config}: OOM")
        return 0.0
    except Exception as e:
        print(f"  Config {config}: Error - {e}")
        return 0.0


def mock_evaluate_attention_config(config: dict[str, Any]) -> float:
    """Mock evaluation for demo mode (no GPU)."""
    import random

    # Heuristics based on typical good configs
    block_m = config.get("block_m", 64)
    block_n = config.get("block_n", 64)
    num_warps = config.get("num_warps", 4)
    num_stages = config.get("num_stages", 2)

    # Score based on heuristics
    # block_m=128, block_n=64, warps=4, stages=3 tends to be good
    score = 100.0

    score += 10 if block_m == 128 else 0
    score += 5 if block_n == 64 else 0
    score += 10 if num_warps == 4 else -5 if num_warps > 8 else 0
    score += 5 if num_stages == 3 else 0

    # Add noise
    score += random.gauss(0, 5)

    return max(0.0, score)


def run_openevolve_tuning():
    """Run OpenEvolve tuning to find optimal attention config."""
    gpu_model = get_nvidia_gpu_model()

    print("=" * 70)
    print("SLM Attention Kernel Tuning with OpenEvolve")
    print("=" * 70)
    print(f"\nGPU: {gpu_model}")
    print(f"Mode: {'MOCK' if MOCK_MODE else 'REAL'}")

    if not HAS_OPENEVOLVE:
        print("\nError: OpenEvolve not installed. Install with: pip install openevolve")
        return None

    # Configuration space for 3D attention kernel
    # block_sizes = [block_b, block_m, block_n]
    config_space = {
        "block_b": [1, 2, 4],  # Batch tile size
        "block_m": [32, 64, 128, 256],  # Sequence M tile size
        "block_n": [32, 64, 128],  # Sequence N tile size
        "num_warps": [2, 4, 8],
        "num_stages": [2, 3, 4],
    }

    print("\nConfiguration space:")
    for param, values in config_space.items():
        print(f"  {param}: {values}")

    # Choose evaluation function
    if MOCK_MODE:
        print("\nRunning in MOCK MODE (no API calls)")
        evaluate_fn = mock_evaluate_attention_config
        max_evals = 30
    else:
        print("\nRunning in REAL MODE (GPU evaluation + OpenAI API)")
        evaluate_fn = evaluate_attention_config
        max_evals = 50

    # Create tuner
    tuner = OpenEvolveTuner(
        config_space=config_space,
        objective=evaluate_fn,
        max_evaluations=max_evals,
        population_size=10,
        temperature=0.3,
        model="gpt-4o-mini",  # Use cheaper model for tuning
        verbose=True,
    )

    print(f"\nStarting OpenEvolve tuning with {max_evals} evaluations...")

    try:
        best_config = tuner.tune()
    except Exception as e:
        print(f"\nTuning failed: {e}")
        print("Falling back to default config...")
        best_config = {
            "block_b": 1,
            "block_m": 128,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 3,
        }

    # Results
    print("\n" + "=" * 70)
    print("OPENEVOLVE TUNING RESULTS")
    print("=" * 70)
    print(f"\nBest configuration found:")
    print(f"  block_sizes = [{best_config['block_b']}, {best_config['block_m']}, {best_config['block_n']}]")
    print(f"  num_warps = {best_config['num_warps']}")
    print(f"  num_stages = {best_config['num_stages']}")

    if not MOCK_MODE and tuner.best_score:
        print(f"  Performance: {tuner.best_score:.2f} TFLOPS")

    # Compare to baseline
    if not MOCK_MODE:
        print("\nComparing to baseline...")
        baseline_config = {
            "block_b": 1,
            "block_m": 64,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 2,
        }
        baseline_tflops = evaluate_attention_config(baseline_config)
        best_tflops = evaluate_attention_config(best_config)

        if baseline_tflops > 0 and best_tflops > 0:
            speedup = best_tflops / baseline_tflops
            print(f"\nBaseline: {baseline_tflops:.2f} TFLOPS")
            print(f"OpenEvolve Best: {best_tflops:.2f} TFLOPS")
            print(f"Speedup: {speedup:.2f}x")

    # Save results
    import json
    results = {
        "gpu": gpu_model,
        "best_config": best_config,
        "best_score": tuner.best_score,
        "evaluations": tuner.evaluation_count,
        "mode": "mock" if MOCK_MODE else "real",
    }

    results_path = "/root/helion/openevolve_attention_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return best_config


def run_grid_search_baseline():
    """Run a grid search baseline to compare against OpenEvolve."""
    print("=" * 70)
    print("Grid Search Baseline for Attention Kernel")
    print("=" * 70)

    # Reduced grid for comparison
    configs_to_test = [
        {"block_b": 1, "block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2},
        {"block_b": 1, "block_m": 128, "block_n": 64, "num_warps": 4, "num_stages": 2},
        {"block_b": 1, "block_m": 128, "block_n": 64, "num_warps": 4, "num_stages": 3},
        {"block_b": 1, "block_m": 64, "block_n": 64, "num_warps": 8, "num_stages": 3},
        {"block_b": 1, "block_m": 128, "block_n": 128, "num_warps": 4, "num_stages": 3},
        {"block_b": 1, "block_m": 256, "block_n": 64, "num_warps": 4, "num_stages": 2},
        {"block_b": 2, "block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2},
        {"block_b": 1, "block_m": 64, "block_n": 32, "num_warps": 4, "num_stages": 3},
    ]

    print(f"\nTesting {len(configs_to_test)} configurations...")

    results = []
    for config in configs_to_test:
        tflops = evaluate_attention_config(config)
        results.append((config, tflops))

    # Sort by performance
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "-" * 70)
    print("Grid Search Results (sorted by TFLOPS):")
    print("-" * 70)
    for config, tflops in results:
        print(f"  [{config['block_b']},{config['block_m']},{config['block_n']}] "
              f"warps={config['num_warps']} stages={config['num_stages']}: "
              f"{tflops:.2f} TFLOPS")

    best_config, best_tflops = results[0]
    print(f"\nBest from grid search: {best_tflops:.2f} TFLOPS")
    print(f"Config: {best_config}")

    return best_config


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--grid":
            run_grid_search_baseline()
        elif sys.argv[1] == "--help":
            print(__doc__)
            print("\nUsage:")
            print("  python slm_openevolve_tuning.py           # Run OpenEvolve tuning")
            print("  python slm_openevolve_tuning.py --grid    # Run grid search baseline")
            print("\nEnvironment:")
            print("  OPENAI_API_KEY    Required for real OpenEvolve tuning")
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        run_openevolve_tuning()


if __name__ == "__main__":
    main()
