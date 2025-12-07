# Helion Attention Kernel Performance Report

**Generated:** 2025-12-07 08:15:44
**GPU:** NVIDIA B200
**Device:** cuda

## Summary

This report compares the performance of the helion attention kernel using various configurations.
The attention kernel is a 3D batched flash attention implementation with online softmax.

### Key Findings

The **Conservative Tuned** configuration (`block_sizes=[1, 128, 64]`, `num_warps=4`, `num_stages=3`)
delivers the best overall performance across most problem sizes, with up to **1.24x speedup** over baseline.

---

## B=2, H=12, S=512, D=64

| Configuration | Time (ms) | TFLOPS | Speedup |
|--------------|-----------|--------|---------|
| Conservative Tuned | 0.0206 | 78.03 | 1.04x |
| Baseline | 0.0215 | 74.79 | baseline |
| B200 Tuned (v1) | 0.1229 | 13.10 | 0.18x |

**Best configuration:** Conservative Tuned

## B=4, H=12, S=1024, D=64

| Configuration | Time (ms) | TFLOPS | Speedup |
|--------------|-----------|--------|---------|
| Baseline | 0.0656 | 196.27 | baseline |
| Conservative Tuned | 0.0666 | 193.49 | 0.99x |
| B200 Tuned (v1) | 0.2548 | 50.56 | 0.26x |

**Best configuration:** Baseline

## B=2, H=16, S=2048, D=64

| Configuration | Time (ms) | TFLOPS | Speedup |
|--------------|-----------|--------|---------|
| Conservative Tuned | 0.1241 | 276.81 | 1.06x |
| Baseline | 0.1322 | 259.92 | baseline |
| B200 Tuned (v1) | 0.9177 | 37.44 | 0.14x |

**Best configuration:** Conservative Tuned

## B=1, H=32, S=4096, D=128

| Configuration | Time (ms) | TFLOPS | Speedup |
|--------------|-----------|--------|---------|
| Conservative Tuned | 0.5745 | 478.47 | 1.24x |
| Baseline | 0.7137 | 385.13 | baseline |
| B200 Tuned (v1) | 50.9329 | 5.40 | 0.01x |

**Best configuration:** Conservative Tuned

---

## Configuration Details

### Baseline Configuration
```python
Config(block_sizes=[1, 64, 64], num_stages=2, num_warps=4)
```
Simple configuration used as reference point.

### Conservative Tuned Configuration (Best Overall)
```python
Config(block_sizes=[1, 128, 64], num_stages=3, num_warps=4)
```
Key optimizations:
- `block_sizes=[1, 128, 64]`: Larger tile sizes in M dimension for better memory coalescing
- `num_warps=4`: Balanced warp count for good occupancy
- `num_stages=3`: Software pipelining for memory latency hiding

### B200 Tuned (v1) Configuration
```python
Config(block_sizes=[1, 64, 64], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], l2_groupings=[8], num_stages=3, num_warps=1, pid_type='xyz', range_flattens=[None, False], range_num_stages=[0, 4])
```
B200-specific features (may require tuning for optimal performance):
- Mixed indexing strategies for different memory access patterns
- xyz pid_type for 3D grid scheduling
- L2 cache grouping for improved locality

---

## End-to-End Transformer Benchmark

This section compares the full transformer performance using helion attention kernel (Conservative Tuned config) vs PyTorch's scaled_dot_product_attention (SDPA).

### Test Configuration
- Model: GPT-2 style transformer (12 heads, 768 embedding dim)
- GPU: NVIDIA B200
- Helion attention kernel config: `block_sizes=[1, 128, 64]`, `num_warps=4`, `num_stages=3`

### Results

| Configuration | Layers | Batch | Seq Len | Helion (ms) | PyTorch SDPA (ms) | Speedup |
|--------------|--------|-------|---------|-------------|-------------------|---------|
| GPT-2 Small | 6 | 2 | 128 | 1.4438 | 1.1895 | 0.82x |
| GPT-2 Medium | 12 | 2 | 128 | 2.8353 | 2.3645 | 0.83x |
| GPT-2 Small, longer seq | 6 | 4 | 256 | 2.4844 | 2.4474 | 0.99x |
| GPT-2 Medium, long seq | 12 | 2 | 512 | 5.0504 | 5.2130 | **1.03x** |

**Average Speedup:** 0.92x

### Analysis

1. **Short Sequences (128):** PyTorch SDPA is faster by ~17-18%. This is expected as PyTorch's flash attention implementation is highly optimized for common transformer workloads.

2. **Medium Sequences (256):** Performance is nearly equal (0.99x), suggesting the helion kernel's efficiency improves with sequence length.

3. **Long Sequences (512):** Helion attention is slightly faster (1.03x), demonstrating the benefit of the flash attention algorithm with larger attention matrices.

### Key Observations

- The helion attention kernel produces **correct results** verified against PyTorch SDPA (within numerical tolerance)
- For the isolated attention kernel benchmark, Conservative Tuned config achieves up to **1.24x speedup** over baseline
- In end-to-end transformer benchmarks, the helion implementation is competitive with PyTorch SDPA
- Performance improves with longer sequences where the flash attention algorithm's O(N) memory usage provides benefits

### Recommendations

1. For **short sequences (< 256)**: PyTorch's SDPA may be more efficient due to its highly optimized implementation
2. For **longer sequences (≥ 512)**: The helion attention kernel becomes competitive and may exceed PyTorch performance
3. For **maximum attention kernel performance**: Use the Conservative Tuned config (`block_sizes=[1, 128, 64]`, `num_warps=4`, `num_stages=3`)
