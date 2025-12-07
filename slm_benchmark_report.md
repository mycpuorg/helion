# Helion Attention Performance Report: Small Language Models

**Generated:** 2025-12-07 09:15:58
**GPU:** NVIDIA B200
**Benchmark:** End-to-end transformer inference (forward pass)


## Helion Overview:

[Helion AutoTuner](https://pytorch.org/wp-content/uploads/2025/10/1-5.png)
[Helion Compiler](https://pytorch.org/wp-content/uploads/2025/10/5-2.png)

## Executive Summary

This report compares the performance of **Helion attention kernels** against **PyTorch's scaled_dot_product_attention (SDPA)** across 10 state-of-the-art small language model architectures.

### Key Results

| Metric | Value |
|--------|-------|
| **Average Speedup** | **1.06x** |
| **Maximum Speedup** | **1.27x** |
| **Minimum Speedup** | 1.00x |
| **Best Model/Config** | Qwen2-0.5B @ seq=4096 |
| **Avg Speedup @ 4096 tokens** | **1.11x** |

### Key Findings

1. **Helion wins across ALL configurations** - Never slower than PyTorch SDPA
2. **Performance scales with sequence length** - Larger speedups at longer sequences
3. **Best results with head_dim=64** - Models like Qwen2-0.5B see up to 1.27x speedup
4. **GQA models work correctly** - All grouped query attention configurations verified

---

## Detailed Results by Model

### Qwen2-0.5B

- **Layers:** 24
- **Query Heads:** 14, **KV Heads:** 2
- **Embed Dim:** 896, **Head Dim:** 64
- **MLP Hidden:** 4864

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 13.61 | 14.33 | 1.05x | +5.3% |
| 1024 | 20.52 | 22.33 | 1.09x | +8.9% |
| 2048 | 36.42 | 43.63 | **1.20x** | +19.8% |
| 4096 | 77.10 | 98.31 | **1.27x** | +27.5% |

### Qwen3-0.6B

- **Layers:** 28
- **Query Heads:** 16, **KV Heads:** 8
- **Embed Dim:** 1024, **Head Dim:** 128
- **MLP Hidden:** 3072

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 15.39 | 15.50 | 1.01x | +0.7% |
| 1024 | 25.97 | 27.58 | 1.06x | +6.2% |
| 2048 | 52.69 | 59.58 | **1.13x** | +13.1% |
| 4096 | 130.31 | 147.43 | **1.13x** | +13.1% |

### Llama-3.2-1B

- **Layers:** 16
- **Query Heads:** 32, **KV Heads:** 8
- **Embed Dim:** 2048, **Head Dim:** 64
- **MLP Hidden:** 8192

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 21.67 | 22.29 | 1.03x | +2.9% |
| 1024 | 41.14 | 43.63 | 1.06x | +6.1% |
| 2048 | 78.90 | 86.12 | 1.09x | +9.2% |
| 4096 | 151.01 | 179.55 | **1.19x** | +18.9% |

### Qwen2-1.5B

- **Layers:** 28
- **Query Heads:** 12, **KV Heads:** 2
- **Embed Dim:** 1536, **Head Dim:** 128
- **MLP Hidden:** 8960

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 33.45 | 33.52 | 1.00x | +0.2% |
| 1024 | 59.85 | 61.61 | 1.03x | +2.9% |
| 2048 | 117.68 | 119.59 | 1.02x | +1.6% |
| 4096 | 234.85 | 256.45 | 1.09x | +9.2% |

### SmolLM2-1.7B

- **Layers:** 24
- **Query Heads:** 32, **KV Heads:** 32
- **Embed Dim:** 2048, **Head Dim:** 64
- **MLP Hidden:** 8192

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 34.06 | 34.99 | 1.03x | +2.7% |
| 1024 | 66.03 | 69.86 | 1.06x | +5.8% |
| 2048 | 128.25 | 139.08 | 1.08x | +8.4% |
| 4096 | 247.45 | 290.98 | **1.18x** | +17.6% |

### Phi-2 (2.7B)

- **Layers:** 32
- **Query Heads:** 32, **KV Heads:** 32
- **Embed Dim:** 2560, **Head Dim:** 80
- **MLP Hidden:** 10240

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 84.81 | 85.09 | 1.00x | +0.3% |
| 1024 | 147.28 | 148.80 | 1.01x | +1.0% |
| 2048 | 292.50 | 298.44 | 1.02x | +2.0% |
| 4096 | 600.31 | 615.17 | 1.02x | +2.5% |

### Qwen2.5-3B

- **Layers:** 36
- **Query Heads:** 16, **KV Heads:** 2
- **Embed Dim:** 2048, **Head Dim:** 128
- **MLP Hidden:** 11008

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 61.16 | 61.20 | 1.00x | +0.1% |
| 1024 | 120.62 | 122.68 | 1.02x | +1.7% |
| 2048 | 219.70 | 228.57 | 1.04x | +4.0% |
| 4096 | 465.01 | 487.09 | 1.05x | +4.7% |

### Llama-3.2-3B

- **Layers:** 28
- **Query Heads:** 24, **KV Heads:** 8
- **Embed Dim:** 3072, **Head Dim:** 128
- **MLP Hidden:** 8192

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 62.98 | 63.81 | 1.01x | +1.3% |
| 1024 | 126.13 | 126.94 | 1.01x | +0.6% |
| 2048 | 237.33 | 247.57 | 1.04x | +4.3% |
| 4096 | 472.46 | 502.22 | 1.06x | +6.3% |

### Phi-3-mini (3.8B)

- **Layers:** 32
- **Query Heads:** 32, **KV Heads:** 32
- **Embed Dim:** 3072, **Head Dim:** 96
- **MLP Hidden:** 8192

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 79.64 | 79.98 | 1.00x | +0.4% |
| 1024 | 163.05 | 164.67 | 1.01x | +1.0% |
| 2048 | 311.18 | 316.81 | 1.02x | +1.8% |
| 4096 | 636.04 | 650.31 | 1.02x | +2.2% |

### rnj-1 (8B)

- **Layers:** 32
- **Query Heads:** 32, **KV Heads:** 8
- **Embed Dim:** 4096, **Head Dim:** 128
- **MLP Hidden:** 16384

| Seq Length | Helion (ms) | PyTorch (ms) | Speedup | Improvement |
|------------|-------------|--------------|---------|-------------|
| 512 | 145.61 | 146.49 | 1.01x | +0.6% |
| 1024 | 295.72 | 299.56 | 1.01x | +1.3% |
| 2048 | 553.27 | 559.12 | 1.01x | +1.1% |
| 4096 | 1170.33 | 1222.69 | 1.04x | +4.5% |

---

## Summary Table: All Models and Sequence Lengths

| Model | 512 | 1024 | 2048 | 4096 | Avg |
|-------|-----|------|------|------|-----|
| Qwen2-0.5B | 1.05x | 1.09x | **1.20x** | **1.27x** | 1.15x |
| Qwen3-0.6B | 1.01x | 1.06x | **1.13x** | **1.13x** | 1.08x |
| Llama-3.2-1B | 1.03x | 1.06x | 1.09x | **1.19x** | 1.09x |
| Qwen2-1.5B | 1.00x | 1.03x | 1.02x | 1.09x | 1.03x |
| SmolLM2-1.7B | 1.03x | 1.06x | 1.08x | **1.18x** | 1.09x |
| Phi-2 | 1.00x | 1.01x | 1.02x | 1.02x | 1.01x |
| Qwen2.5-3B | 1.00x | 1.02x | 1.04x | 1.05x | 1.03x |
| Llama-3.2-3B | 1.01x | 1.01x | 1.04x | 1.06x | 1.03x |
| Phi-3-mini | 1.00x | 1.01x | 1.02x | 1.02x | 1.01x |
| rnj-1 | 1.01x | 1.01x | 1.01x | 1.04x | 1.02x |

---

## Performance Analysis

### By Sequence Length

| Sequence Length | Avg Speedup | Min | Max |
|-----------------|-------------|-----|-----|
| 512 | 1.01x | 1.00x | 1.05x |
| 1024 | 1.04x | 1.01x | 1.09x |
| 2048 | 1.07x | 1.01x | 1.20x |
| 4096 | 1.11x | 1.02x | 1.27x |

### By Head Dimension

| Head Dim | Avg Speedup | Models |
|----------|-------------|--------|
| 64 | 1.11x | Qwen2-0.5B, Llama-3.2-1B, SmolLM2-1.7B |
| 80 | 1.01x | Phi-2 |
| 96 | 1.01x | Phi-3-mini |
| 128 | 1.04x | Qwen3-0.6B, Qwen2-1.5B, Qwen2.5-3B (+2 more) |

---

## Methodology

- **Hardware:** NVIDIA B200
- **Benchmark Type:** End-to-end transformer forward pass
- **Batch Size:** 1
- **Warmup:** 3 iterations
- **Measurement:** Interleaved benchmarking for fair comparison
- **Correctness:** All configurations verified against PyTorch reference (rtol=1e-2, atol=1e-2)

### Helion Attention Configuration

The helion attention kernel uses flash attention with online softmax. Different configurations are used based on head dimension:

| Head Dim | Block Sizes | Num Warps | Num Stages |
|----------|-------------|-----------|------------|
| 64 | [1, 128, 64] | 4 | 3 |
| 80 | [1, 64, 64] | 4 | 2 |
| 96 | [1, 64, 64] | 4 | 2 |
| 128 | [1, 64, 32] | 4 | 2 |

### OpenEvolve Autotuner Integration

The Helion framework includes an **OpenEvolve-based autotuner** (`helion.autotuner.openevolve_tuner.OpenEvolveTuner`) that uses evolutionary algorithms powered by LLMs to optimize kernel configurations.

#### Tuning Results (head_dim=64, seq_len=1024)

| Configuration | Block Sizes | Warps | Stages | TFLOPS | Speedup |
|--------------|-------------|-------|--------|--------|---------|
| **Tuned (Best)** | [1, 128, 64] | 4 | 3 | **94.43** | **1.20x** |
| Baseline | [1, 64, 64] | 4 | 2 | 78.93 | 1.00x |
| Conservative | [1, 128, 64] | 4 | 2 | 82.21 | 1.04x |
| High Warps | [1, 64, 64] | 8 | 3 | 78.51 | 0.99x |

The tuned configuration (`block_sizes=[1, 128, 64], num_warps=4, num_stages=3`) delivers **19.6% improvement** over the baseline, which is used throughout this benchmark.

#### Using the OpenEvolve Tuner

```python
from helion.autotuner.openevolve_tuner import OpenEvolveTuner

config_space = {
    "block_b": [1, 2, 4],
    "block_m": [32, 64, 128, 256],
    "block_n": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}

tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=your_benchmark_function,  # Returns TFLOPS
    max_evaluations=50,
    model="gpt-4o-mini",
)

best_config = tuner.tune()
```

---

## Model Sources

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) - Alibaba
- [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) - Alibaba
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) - Meta
- [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) - Alibaba
- [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) - Hugging Face
- [Phi-2](https://huggingface.co/microsoft/phi-2) - Microsoft
- [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) - Alibaba
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) - Meta
- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) - Microsoft
- [rnj-1](https://huggingface.co/EssentialAI/rnj-1) - Essential AI

---

*Report generated by helion SLM benchmark suite*
