# LLM-First Parallel Autotuner Design

## Overview

This document describes an LLM-driven autotuning approach that puts the language model in the driver's seat for GPU kernel configuration search, leveraging Helion's parallel benchmarking infrastructure for fast evaluation.

**Core insight**: The LLM's unique value is intelligent exploration and domain reasoning. Parallel benchmarking makes evaluations cheap. Therefore, let the LLM decide what to benchmark, and benchmark everything it suggests in parallel.

---

## Design Philosophy

### Why LLM-First?

| Component | Role |
|-----------|------|
| **LLM** | The brain - decides what configs to try based on domain knowledge |
| **Parallel Benchmark** | The muscle - evaluates configs fast via parallelization |
| **Feedback Loop** | The learning - LLM sees real results, improves each round |

### What LLM Brings That Others Can't

1. **Domain reasoning**: "block_size=256 with num_warps=8 will cause register spill"
2. **Pattern recognition**: "Configs with num_stages>2 performed well, explore that"
3. **Intelligent jumps**: Can suggest radically different configs, not just neighbors
4. **Hardware awareness**: Can reason about occupancy, memory coalescing, cache behavior

### Why Not Surrogate Filtering?

Surrogate models (Random Forest) were designed to reduce benchmark costs. But:

- With parallel benchmarking, evaluating 50 configs takes ~same time as 10
- Surrogates can filter out good configs (false negatives)
- Surrogates lack domain knowledge that LLM has
- Added complexity without clear quality benefit

**Conclusion**: Skip the surrogate. Let LLM decide, benchmark everything in parallel.

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROUND 1: INITIAL EXPLORATION                    │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   LLM generates  │───▶│ Parallel bench   │───▶│  Collect results │  │
│  │   20-30 diverse  │    │ all configs      │    │  + rank by perf  │  │
│  │   configs        │    │ (~30s)           │    │                  │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                         │
│  LLM prompt includes:                                                   │
│  • Config space definition (parameters, valid ranges)                   │
│  • GPU hardware info (type, memory, compute units)                      │
│  • Kernel characteristics (memory-bound vs compute-bound hints)         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                      ROUNDS 2-N: ITERATIVE REFINEMENT                   │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   LLM analyzes   │───▶│ Parallel bench   │───▶│  Update history  │  │
│  │   history, gens  │    │ new configs      │    │  + check conv.   │  │
│  │   next 15-20     │    │                  │    │                  │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                         │
│  LLM prompt includes:                                                   │
│  • Full history: all configs tried + their measured performance         │
│  • Analysis: what patterns worked, what failed                          │
│  • Current best config and its performance                              │
│  • Request: generate configs that might beat the current best           │
│                                                                         │
│                              ↓                                          │
│                    Repeat until convergence                             │
│                    (typically 3-5 rounds)                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                         FINAL: BEST CONFIG                              │
│                                                                         │
│  Return config with lowest measured latency across all rounds           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Round Details

### Round 1: Initial Exploration

**Goal**: Get diverse coverage of the config space with intelligent starting points.

**LLM Prompt Structure**:
```
You are optimizing a GPU kernel configuration for maximum performance.

HARDWARE:
- GPU: {gpu_name} ({gpu_arch})
- Compute units: {num_cus}
- Memory: {memory_gb} GB, {memory_bandwidth} GB/s
- Max threads per block: {max_threads}

KERNEL CHARACTERISTICS:
- Type: {kernel_type} (e.g., matmul, attention, reduction)
- Estimated arithmetic intensity: {ai_estimate}
- Memory access pattern: {access_pattern}

CONFIG SPACE:
{config_space_json}

Generate {batch_size} diverse configurations to benchmark.
Include:
- Conservative configs (safe defaults)
- Aggressive configs (push limits)
- Exploratory configs (unusual combinations)

Consider:
- Memory coalescing: larger block sizes for aligned access
- Occupancy: balance num_warps vs register pressure
- Pipelining: num_stages helps memory-bound kernels
- Block shape: match data access patterns

Return JSON:
{
  "configs": [
    {"reasoning": "why this config", "config": {...}},
    ...
  ]
}
```

**Output**: 20-30 configs with LLM's reasoning for each.

**Benchmark**: All configs evaluated in parallel using Helion's infrastructure.

---

### Rounds 2-N: Iterative Refinement

**Goal**: Learn from results, focus on promising regions, escape local optima.

**LLM Prompt Structure**:
```
You are optimizing a GPU kernel. Here's what we've learned so far.

HARDWARE: {same as round 1}

FULL HISTORY (sorted by performance):
{history_table}
| Rank | Config | Latency (ms) | Status |
|------|--------|--------------|--------|
| 1    | {...}  | 0.82         | best   |
| 2    | {...}  | 0.89         | good   |
| ...  | {...}  | 1.45         | ok     |
| N    | {...}  | inf          | failed |

OBSERVATIONS:
- Best config so far: {best_config} at {best_latency}ms
- Patterns that worked: {worked_patterns}
- Patterns that failed: {failed_patterns}
- Unexplored regions: {unexplored}

Generate {batch_size} new configurations.
Strategy:
- Exploit: Variations of top 3 configs
- Explore: Try unexplored regions
- Escape: If stuck, try radically different configs

Return JSON:
{
  "analysis": "Your analysis of the results so far",
  "strategy": "Your strategy for this round",
  "configs": [
    {"reasoning": "why this config", "config": {...}},
    ...
  ]
}
```

**Output**: 15-20 configs with reasoning and strategy explanation.

**Convergence Check**: Stop if no improvement for 2 consecutive rounds, or max rounds reached.

---

## Convergence Criteria

The search stops when any of these conditions are met:

1. **No improvement**: Best config unchanged for 2 consecutive rounds
2. **Max rounds**: Reached maximum rounds (default: 5)
3. **Target achieved**: Performance meets user-specified target
4. **Diminishing returns**: Improvement < 1% over last 2 rounds

---

## Configuration Options

```python
@dataclass
class LLMAutotunerConfig:
    # LLM settings
    model: str = "gpt-4o"                    # or "claude-sonnet-4-20250514", local models
    temperature: float = 0.7                  # higher = more exploration
    api_base: str | None = None              # for local/custom endpoints

    # Search parameters
    initial_batch_size: int = 25             # configs in round 1
    refinement_batch_size: int = 15          # configs in rounds 2+
    max_rounds: int = 5                      # maximum iterations
    patience: int = 2                        # rounds without improvement before stopping
    min_improvement: float = 0.01            # 1% improvement threshold

    # Parallelization
    use_parallel_benchmark: bool = True
    precompile_mode: str = "fork"            # "fork", "spawn", or None

    # Prompt customization
    include_kernel_source: bool = False      # include Helion source in prompt
    include_triton_hints: bool = True        # include Triton-specific guidance
    hardware_context: bool = True            # include GPU hardware details

    # Fallback
    fallback_on_llm_failure: str = "random"  # "random", "pattern", or "error"
```

---

## Expected Performance

### Benchmark Budget

| Round | Configs | LLM Calls | Wall-Clock |
|-------|---------|-----------|------------|
| 1     | 25      | 1         | ~30s       |
| 2     | 15      | 1         | ~25s       |
| 3     | 15      | 1         | ~25s       |
| 4     | 15      | 1         | ~25s       |
| 5     | 15      | 1         | ~25s       |
| **Total** | **~85** | **~5** | **~2-3 min** |

### Comparison

| Approach | Configs | LLM Calls | Wall-Clock | Intelligence |
|----------|---------|-----------|------------|--------------|
| Random Search | 200 | 0 | ~3 min | None |
| Pattern Search | 150 | 0 | ~2 min | Local only |
| LFBO | 250 | 0 | ~3 min | Surrogate |
| OpenEvolve | 100 | 100 | ~5 min | Full LLM |
| **LLM-First** | **85** | **5** | **~2 min** | **Full LLM** |

### Quality Expectations

- **vs Random**: Significantly better (LLM finds good regions faster)
- **vs Pattern Search**: Better (LLM can escape local optima)
- **vs LFBO**: Similar or better (LLM has domain knowledge)
- **vs OpenEvolve**: Similar quality, 20x faster wall-clock

---

## Implementation Plan

### Integration with Helion

```python
# File: helion/autotuner/llm_search.py

class LLMSearch(PopulationBasedSearch):
    """LLM-driven autotuner with parallel benchmarking."""

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        config: LLMAutotunerConfig = None,
    ):
        super().__init__(kernel, args)
        self.config = config or LLMAutotunerConfig()
        self.llm_client = self._init_llm_client()
        self.history: list[tuple[Config, float]] = []

    def _autotune(self) -> Config:
        # Round 1: Initial exploration
        configs = self._llm_generate_initial()
        results = self.parallel_benchmark([c for c in configs], desc="Round 1")
        self._update_history(results)

        # Rounds 2-N: Iterative refinement
        for round_num in range(2, self.config.max_rounds + 1):
            if self._should_stop():
                break

            configs = self._llm_generate_refinement(round_num)
            results = self.parallel_benchmark(configs, desc=f"Round {round_num}")
            self._update_history(results)

        return self._get_best_config()
```

### LLM Client Interface

```python
class LLMClient(Protocol):
    """Interface for LLM providers."""

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        response_format: dict | None = None,  # for JSON mode
    ) -> str:
        ...

# Implementations:
# - OpenAIClient (GPT-4, GPT-4o, GPT-4o-mini)
# - AnthropicClient (Claude Opus, Sonnet)
# - OllamaClient (local models)
# - OpenRouterClient (any model via OpenRouter)
```

### Prompt Builder

```python
class PromptBuilder:
    """Builds prompts for LLM autotuning."""

    def __init__(self, kernel: BoundKernel, config_spec: ConfigSpec):
        self.kernel = kernel
        self.config_spec = config_spec
        self.hardware_info = self._get_hardware_info()

    def build_initial_prompt(self, batch_size: int) -> str:
        """Build prompt for round 1."""
        ...

    def build_refinement_prompt(
        self,
        history: list[tuple[Config, float]],
        round_num: int,
        batch_size: int,
    ) -> str:
        """Build prompt for rounds 2+."""
        ...

    def _get_hardware_info(self) -> dict:
        """Get GPU hardware details."""
        ...

    def _format_config_space(self) -> str:
        """Format config space for LLM."""
        ...
```

---

## Prompt Engineering Details

### Config Space Formatting

Present the config space in a clear, structured way:

```json
{
  "block_sizes": {
    "description": "Block dimensions for tiling",
    "type": "list of powers of 2",
    "constraints": "product <= 1024 (max threads per block)",
    "dimensions": ["M", "N", "K"],
    "valid_values": [16, 32, 64, 128, 256]
  },
  "num_warps": {
    "description": "Number of warps (32 threads each)",
    "type": "power of 2",
    "range": [1, 32],
    "tradeoff": "more warps = better latency hiding, but more register pressure"
  },
  "num_stages": {
    "description": "Software pipelining stages",
    "type": "integer",
    "range": [1, 8],
    "tradeoff": "more stages = better memory latency hiding, but more shared memory"
  }
}
```

### History Formatting

Present history as a ranked table with insights:

```
PERFORMANCE HISTORY (85 configs evaluated):

TOP 5:
| Rank | block_m | block_n | block_k | warps | stages | Latency |
|------|---------|---------|---------|-------|--------|---------|
| 1    | 128     | 128     | 32      | 4     | 3      | 0.82ms  |
| 2    | 128     | 64      | 32      | 4     | 3      | 0.89ms  |
| 3    | 64      | 128     | 32      | 4     | 2      | 0.91ms  |
| 4    | 128     | 128     | 64      | 8     | 2      | 0.95ms  |
| 5    | 64      | 64      | 32      | 4     | 3      | 0.98ms  |

FAILED CONFIGS (compiled but crashed or wrong results): 12
TIMED OUT (compilation hung): 3

PATTERNS OBSERVED:
- block_k=32 appears in all top 5 (sweet spot for this kernel)
- num_stages >= 2 helps (memory-bound kernel)
- num_warps=4 dominates (register pressure at warps=8)
- block_m=block_n=128 is best, but 64 also works
```

---

## Error Handling

### LLM Failures

```python
def _llm_generate_with_fallback(self, prompt: str) -> list[Config]:
    try:
        response = self.llm_client.generate(prompt)
        configs = self._parse_configs(response)
        return self._validate_configs(configs)
    except LLMError as e:
        self.log.warning(f"LLM failed: {e}, using fallback")
        return self._fallback_generate()

def _fallback_generate(self) -> list[Config]:
    match self.config.fallback_on_llm_failure:
        case "random":
            return [self.config_gen.random() for _ in range(self.config.refinement_batch_size)]
        case "pattern":
            return self._pattern_neighbors(self._get_best_config())
        case "error":
            raise
```

### Invalid Configs

LLM may generate invalid configs. Handle gracefully:

```python
def _validate_configs(self, configs: list[dict]) -> list[Config]:
    valid = []
    for cfg_dict in configs:
        try:
            config = self.config_spec.from_dict(cfg_dict)
            if self.config_spec.is_valid(config):
                valid.append(config)
            else:
                self.log.debug(f"Invalid config from LLM: {cfg_dict}")
        except Exception as e:
            self.log.debug(f"Failed to parse LLM config: {e}")

    if len(valid) < len(configs) * 0.5:
        self.log.warning(f"LLM generated many invalid configs: {len(valid)}/{len(configs)}")

    return valid
```

---

## Future Enhancements

1. **Streaming responses**: Start benchmarking configs as LLM generates them
2. **Multi-GPU**: Distribute benchmarks across multiple GPUs
3. **Caching**: Cache LLM responses for similar config spaces
4. **Fine-tuning**: Train specialized model on autotuning data
5. **Hybrid on-demand**: Only invoke surrogate if LLM is unavailable
6. **Cross-kernel transfer**: Use insights from similar kernels

---

## References

- OpenEvolve: https://github.com/codelion/openevolve
- Helion parallel benchmarking: `helion/autotuner/base_search.py`
- AlphaEvolve (DeepMind): LLM-guided code optimization
