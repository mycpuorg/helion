"""
LLM-First Parallel Autotuner.

This autotuner puts the LLM in the driver's seat for GPU kernel configuration search,
leveraging Helion's parallel benchmarking infrastructure for fast evaluation.

Core insight: The LLM's unique value is intelligent exploration and domain reasoning.
Parallel benchmarking makes evaluations cheap. Therefore, let the LLM decide what to
benchmark, and benchmark everything it suggests in parallel.

Usage:
    HELION_AUTOTUNER=LLMSearch python your_script.py

Environment variables:
    HELION_LLM_MODEL: LLM model to use (default: "gpt-4o")
    HELION_LLM_API_BASE: API base URL (default: OpenAI)
    HELION_LLM_INITIAL_BATCH: Initial batch size (default: 25)
    HELION_LLM_REFINEMENT_BATCH: Refinement batch size (default: 15)
    HELION_LLM_MAX_ROUNDS: Maximum rounds (default: 5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from .. import exc
from ..runtime.config import Config
from .base_search import BaseSearch
from .base_search import BenchmarkResult
from .config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel


logger = logging.getLogger(__name__)


@dataclass
class LLMSearchConfig:
    """Configuration for LLM-based autotuning."""

    # LLM settings
    model: str = "gpt-4o"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7

    # Search parameters
    initial_batch_size: int = 25
    refinement_batch_size: int = 15
    max_rounds: int = 5
    patience: int = 2  # Rounds without improvement before stopping
    min_improvement: float = 0.01  # 1% improvement threshold

    # Fallback
    fallback_on_llm_failure: str = "random"  # "random", "pattern", or "error"

    @classmethod
    def from_env(cls) -> LLMSearchConfig:
        """Create config from environment variables."""
        return cls(
            model=os.environ.get("HELION_LLM_MODEL", "gpt-4o"),
            api_base=os.environ.get("HELION_LLM_API_BASE"),
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
            temperature=float(os.environ.get("HELION_LLM_TEMPERATURE", "0.7")),
            initial_batch_size=int(os.environ.get("HELION_LLM_INITIAL_BATCH", "25")),
            refinement_batch_size=int(os.environ.get("HELION_LLM_REFINEMENT_BATCH", "15")),
            max_rounds=int(os.environ.get("HELION_LLM_MAX_ROUNDS", "5")),
            patience=int(os.environ.get("HELION_LLM_PATIENCE", "2")),
            min_improvement=float(os.environ.get("HELION_LLM_MIN_IMPROVEMENT", "0.01")),
        )


@dataclass
class HistoryEntry:
    """A single entry in the tuning history."""

    config: dict[str, Any]
    latency_ms: float
    status: str  # "ok", "error", "timeout"
    round_num: int


@dataclass
class TuningHistory:
    """Tracks all configurations tried during tuning."""

    entries: list[HistoryEntry] = field(default_factory=list)

    def add(
        self,
        config: Config,
        latency_ms: float,
        status: str,
        round_num: int,
    ) -> None:
        """Add a new entry to history."""
        self.entries.append(
            HistoryEntry(
                config=dict(config.config),
                latency_ms=latency_ms,
                status=status,
                round_num=round_num,
            )
        )

    def get_best(self) -> HistoryEntry | None:
        """Get the best entry (lowest latency among successful runs)."""
        ok_entries = [e for e in self.entries if e.status == "ok" and math.isfinite(e.latency_ms)]
        if not ok_entries:
            return None
        return min(ok_entries, key=lambda e: e.latency_ms)

    def get_top_k(self, k: int) -> list[HistoryEntry]:
        """Get top k entries by latency."""
        ok_entries = [e for e in self.entries if e.status == "ok" and math.isfinite(e.latency_ms)]
        return sorted(ok_entries, key=lambda e: e.latency_ms)[:k]

    def format_for_llm(self, top_k: int = 10) -> str:
        """Format history for LLM prompt."""
        lines = []
        lines.append(f"PERFORMANCE HISTORY ({len(self.entries)} configs evaluated):")
        lines.append("")

        top_entries = self.get_top_k(top_k)
        if top_entries:
            lines.append(f"TOP {len(top_entries)}:")
            for i, entry in enumerate(top_entries, 1):
                config_str = json.dumps(entry.config, indent=None)
                lines.append(f"  {i}. {entry.latency_ms:.4f}ms - {config_str}")
            lines.append("")

        # Count failures
        failed = sum(1 for e in self.entries if e.status == "error")
        timeout = sum(1 for e in self.entries if e.status == "timeout")
        if failed or timeout:
            lines.append(f"FAILED: {failed} configs, TIMEOUT: {timeout} configs")
            lines.append("")

        best = self.get_best()
        if best:
            lines.append(f"BEST SO FAR: {best.latency_ms:.4f}ms")
            lines.append(f"BEST CONFIG: {json.dumps(best.config, indent=2)}")

        return "\n".join(lines)


class LLMClient:
    """Client for communicating with LLM APIs."""

    def __init__(self, config: LLMSearchConfig) -> None:
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of the client."""
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as e:
            raise exc.AutotuneError(
                "LLMSearch requires the openai package. Install with: pip install openai"
            ) from e

        api_key = self.config.api_key
        if not api_key:
            raise exc.AutotuneError(
                "LLMSearch requires an API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base

        self._client = openai.OpenAI(**kwargs)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            raise


class PromptBuilder:
    """Builds prompts for LLM autotuning."""

    def __init__(self, kernel: BoundKernel, config_gen: Any) -> None:
        self.kernel = kernel
        self.config_gen = config_gen
        self.config_spec = kernel.config_spec

    def _get_hardware_info(self) -> dict[str, Any]:
        """Get GPU hardware information."""
        import torch

        if not torch.cuda.is_available():
            return {"gpu": "Unknown", "compute_capability": "Unknown"}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
        }

    def _format_config_space(self) -> str:
        """Format the configuration space for the LLM."""
        space_desc = []

        # Get default config to understand structure
        default_config = self.config_spec.default_config()
        config_dict = default_config.config

        for key, value in config_dict.items():
            if key == "block_sizes":
                space_desc.append({
                    "name": "block_sizes",
                    "description": "Block dimensions for tiling (list of powers of 2)",
                    "current_default": value,
                    "valid_values": "Powers of 2: 16, 32, 64, 128, 256, 512, 1024",
                    "constraints": "Product should not exceed max threads per block",
                })
            elif key == "num_warps":
                space_desc.append({
                    "name": "num_warps",
                    "description": "Number of warps (32 threads each)",
                    "current_default": value,
                    "valid_values": [1, 2, 4, 8, 16, 32],
                    "tradeoff": "More warps = better latency hiding, but more register pressure",
                })
            elif key == "num_stages":
                space_desc.append({
                    "name": "num_stages",
                    "description": "Software pipelining stages",
                    "current_default": value,
                    "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
                    "tradeoff": "More stages = better memory latency hiding, but more shared memory",
                })
            elif key == "pid_type":
                space_desc.append({
                    "name": "pid_type",
                    "description": "Block scheduling strategy",
                    "current_default": value,
                    "valid_values": ["flat", "xyz", "persistent_blocked", "persistent_interleaved"],
                })
            elif key == "indexing":
                space_desc.append({
                    "name": "indexing",
                    "description": "Pointer indexing mode",
                    "current_default": value,
                    "valid_values": ["pointer", "block_ptr", "tensor_descriptor"],
                })
            else:
                # Generic handling for other parameters
                space_desc.append({
                    "name": key,
                    "current_default": value,
                })

        return json.dumps(space_desc, indent=2)

    def build_system_prompt(self) -> str:
        """Build the system prompt."""
        return """You are an expert GPU kernel optimizer specializing in Triton kernels.
Your task is to generate configurations that maximize kernel performance (minimize latency).

KEY PRINCIPLES:
1. Memory coalescing: Larger block sizes help with aligned memory access
2. Occupancy: Balance num_warps against register pressure
3. Pipelining: num_stages helps hide memory latency for memory-bound kernels
4. Block scheduling: persistent modes help with small problem sizes

IMPORTANT RULES:
- block_sizes must be powers of 2 (16, 32, 64, 128, 256, 512, 1024)
- num_warps must be a power of 2 (1, 2, 4, 8, 16, 32)
- The product of block_sizes should not exceed 1024 (max threads per block)
- Always return valid JSON with the exact structure requested

Be creative but practical. Generate diverse configs that explore different tradeoffs."""

    def build_initial_prompt(self, batch_size: int) -> str:
        """Build prompt for round 1 (initial exploration)."""
        hw_info = self._get_hardware_info()
        config_space = self._format_config_space()

        return f"""Generate {batch_size} diverse GPU kernel configurations for benchmarking.

HARDWARE:
{json.dumps(hw_info, indent=2)}

CONFIGURATION SPACE:
{config_space}

Generate a mix of:
- Conservative configs (safe defaults, moderate values)
- Aggressive configs (push limits for maximum throughput)
- Exploratory configs (unusual combinations to discover surprises)

Return JSON in this exact format:
{{
  "configs": [
    {{
      "reasoning": "Brief explanation of why this config might work well",
      "config": {{
        "block_sizes": [64],
        "num_warps": 4,
        "num_stages": 2
      }}
    }},
    ...more configs...
  ]
}}

Generate exactly {batch_size} configs. Each config must have valid values."""

    def build_refinement_prompt(
        self,
        history: TuningHistory,
        round_num: int,
        batch_size: int,
    ) -> str:
        """Build prompt for rounds 2+ (iterative refinement)."""
        hw_info = self._get_hardware_info()
        config_space = self._format_config_space()
        history_str = history.format_for_llm(top_k=10)

        best = history.get_best()
        best_config_str = json.dumps(best.config, indent=2) if best else "None found yet"

        return f"""Round {round_num}: Generate {batch_size} new configurations based on what we've learned.

HARDWARE:
{json.dumps(hw_info, indent=2)}

CONFIGURATION SPACE:
{config_space}

{history_str}

CURRENT BEST CONFIG:
{best_config_str}

STRATEGY FOR THIS ROUND:
1. EXPLOIT: Generate variations of the top 3 configs (small perturbations)
2. EXPLORE: Try regions we haven't explored yet
3. ESCAPE: If we seem stuck, try radically different configurations

Return JSON in this exact format:
{{
  "analysis": "Your analysis of patterns in the results so far",
  "strategy": "Your strategy for this round's configs",
  "configs": [
    {{
      "reasoning": "Why this config might beat the current best",
      "config": {{
        "block_sizes": [64],
        "num_warps": 4,
        "num_stages": 2
      }}
    }},
    ...more configs...
  ]
}}

Generate exactly {batch_size} configs. Focus on beating {best.latency_ms:.4f}ms."""


class LLMSearch(BaseSearch):
    """
    LLM-driven autotuner with parallel benchmarking.

    This autotuner uses an LLM to intelligently generate configurations,
    then benchmarks them in parallel. It iterates through multiple rounds,
    feeding results back to the LLM to refine its suggestions.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        config: LLMSearchConfig | None = None,
    ) -> None:
        super().__init__(kernel, args)
        self.llm_config = config or LLMSearchConfig.from_env()
        self.llm_client = LLMClient(self.llm_config)
        self.config_gen = ConfigGeneration(self.config_spec)
        self.prompt_builder = PromptBuilder(kernel, self.config_gen)
        self.history = TuningHistory()
        self._best_perf_by_round: list[float] = []
        self._current_generation: int = 0

    def set_generation(self, generation: int) -> None:
        """Set the current generation/round number."""
        self._current_generation = generation

    def _parse_llm_configs(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response and extract configurations."""
        try:
            data = json.loads(response)
            configs = data.get("configs", [])
            return [c.get("config", c) for c in configs if isinstance(c, dict)]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract JSON from response
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                try:
                    data = json.loads(match.group())
                    configs = data.get("configs", [])
                    return [c.get("config", c) for c in configs if isinstance(c, dict)]
                except json.JSONDecodeError:
                    pass
            return []

    def _dict_to_config(self, config_dict: dict[str, Any]) -> Config | None:
        """Convert a dictionary to a Config object, validating it."""
        try:
            # Start with default config and update with LLM suggestions
            default = self.config_spec.default_config()
            merged = dict(default.config)

            # Update with LLM-provided values
            for key, value in config_dict.items():
                if key in merged:
                    merged[key] = value

            # Create and normalize config
            config = Config(**merged)
            self.config_spec.normalize(config.config)

            # Validate by attempting to generate code (catches structural issues)
            try:
                self.kernel.to_triton_code(config, emit_repro_caller=False)
            except Exception as e:
                logger.debug(f"Config validation failed (code generation): {e}")
                return None

            return config
        except Exception as e:
            logger.debug(f"Failed to convert config dict to Config: {e}")
            return None

    def _llm_generate_configs(
        self,
        round_num: int,
        batch_size: int,
    ) -> list[Config]:
        """Generate configs using LLM."""
        try:
            system_prompt = self.prompt_builder.build_system_prompt()

            if round_num == 1:
                user_prompt = self.prompt_builder.build_initial_prompt(batch_size)
            else:
                user_prompt = self.prompt_builder.build_refinement_prompt(
                    self.history, round_num, batch_size
                )

            self.log(f"Round {round_num}: Requesting {batch_size} configs from LLM...")
            response = self.llm_client.generate(system_prompt, user_prompt)

            config_dicts = self._parse_llm_configs(response)
            self.log(f"Round {round_num}: LLM returned {len(config_dicts)} config suggestions")

            # Convert to Config objects
            configs = []
            for cd in config_dicts:
                config = self._dict_to_config(cd)
                if config is not None:
                    configs.append(config)

            self.log(f"Round {round_num}: {len(configs)} valid configs after parsing")
            return configs

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._fallback_generate(batch_size)

    def _fallback_generate(self, count: int) -> list[Config]:
        """Generate configs using fallback method when LLM fails."""
        fallback = self.llm_config.fallback_on_llm_failure

        if fallback == "error":
            raise exc.AutotuneError("LLM generation failed and fallback is set to 'error'")

        self.log(f"Using fallback method: {fallback}")

        # Always use random generation for simplicity
        configs = []
        for _ in range(count):
            config = self.config_gen.random_config()
            configs.append(config)
        return configs

    def _benchmark_configs(
        self,
        configs: list[Config],
        round_num: int,
        desc: str,
    ) -> list[BenchmarkResult]:
        """Benchmark a list of configs and update history."""
        # Use parallel_benchmark directly with Config objects
        results = self.parallel_benchmark(configs, desc=desc)

        # Update history
        for result in results:
            self.history.add(
                config=result.config,
                latency_ms=result.perf,
                status=result.status,
                round_num=round_num,
            )

        return results

    def _should_stop(self, round_num: int) -> bool:
        """Check if we should stop searching."""
        if round_num > self.llm_config.max_rounds:
            self.log("Stopping: reached max rounds")
            return True

        if len(self._best_perf_by_round) < 2:
            return False

        # Check for improvement
        rounds_without_improvement = 0
        for i in range(len(self._best_perf_by_round) - 1, 0, -1):
            prev = self._best_perf_by_round[i - 1]
            curr = self._best_perf_by_round[i]
            improvement = (prev - curr) / prev if prev > 0 else 0
            if improvement < self.llm_config.min_improvement:
                rounds_without_improvement += 1
            else:
                break

        if rounds_without_improvement >= self.llm_config.patience:
            self.log(f"Stopping: no improvement for {rounds_without_improvement} rounds")
            return True

        return False

    def _autotune(self) -> Config:
        """
        Run LLM-first parallel autotuning.

        Returns:
            The best configuration found.
        """
        self.log("=" * 70)
        self.log("LLM-First Parallel Autotuner")
        self.log("=" * 70)
        self.log(f"Model: {self.llm_config.model}")
        self.log(f"Initial batch: {self.llm_config.initial_batch_size}")
        self.log(f"Refinement batch: {self.llm_config.refinement_batch_size}")
        self.log(f"Max rounds: {self.llm_config.max_rounds}")
        self.log("=" * 70)

        start_time = time.perf_counter()

        # Round 1: Initial exploration
        round_num = 1
        self.set_generation(round_num)
        configs = self._llm_generate_configs(round_num, self.llm_config.initial_batch_size)

        if not configs:
            self.log("Warning: LLM returned no configs, using fallback")
            configs = self._fallback_generate(self.llm_config.initial_batch_size)

        self._benchmark_configs(configs, round_num, f"Round {round_num}")

        best = self.history.get_best()
        if best:
            self._best_perf_by_round.append(best.latency_ms)
            self.log(f"Round {round_num} complete: best = {best.latency_ms:.4f}ms")
        else:
            self._best_perf_by_round.append(float('inf'))
            self.log(f"Round {round_num} complete: no successful configs")

        # Rounds 2-N: Iterative refinement
        for round_num in range(2, self.llm_config.max_rounds + 1):
            if self._should_stop(round_num):
                break

            self.set_generation(round_num)
            configs = self._llm_generate_configs(round_num, self.llm_config.refinement_batch_size)

            if not configs:
                self.log(f"Round {round_num}: LLM returned no configs, using fallback")
                configs = self._fallback_generate(self.llm_config.refinement_batch_size)

            self._benchmark_configs(configs, round_num, f"Round {round_num}")

            best = self.history.get_best()
            if best:
                self._best_perf_by_round.append(best.latency_ms)
                self.log(f"Round {round_num} complete: best = {best.latency_ms:.4f}ms")
            else:
                self._best_perf_by_round.append(self._best_perf_by_round[-1])
                self.log(f"Round {round_num} complete: no improvement")

        # Get final best
        elapsed = time.perf_counter() - start_time
        best_entry = self.history.get_best()

        self.log("=" * 70)
        if best_entry:
            self.log(f"Best config found: {best_entry.latency_ms:.4f}ms")
            self.log(f"Total configs evaluated: {len(self.history.entries)}")
            self.log(f"Total time: {elapsed:.1f}s")
            self.log("=" * 70)

            # Return the best config
            return Config(**best_entry.config)
        else:
            # No successful configs - return default
            self.log("Warning: No successful configs found, returning default")
            self.log("=" * 70)
            return self.config_spec.default_config()

    @property
    def statistics(self) -> str:
        """Return statistics about the search."""
        total = len(self.history.entries)
        ok = sum(1 for e in self.history.entries if e.status == "ok")
        best = self.history.get_best()
        best_str = f"{best.latency_ms:.4f}ms" if best else "N/A"
        return f"total={total} ok={ok} best={best_str}"
