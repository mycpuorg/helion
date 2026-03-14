"""
LLM-Seeded Warm Start for Autotuning (Path B Fallback).

This implements the fallback approach from the hackathon plan:
After the initial random phase, serialize top-20 configs + their latencies,
prompt an LLM for 10 novel configs exploiting patterns, inject those as
additional search seeds.

This demonstrates EvoX's insight that the selection mechanism matters.
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LLMSuggestion:
    """A configuration suggested by the LLM."""
    config: Dict[str, Any]
    reasoning: str
    confidence: float


class LLMWarmStartStrategy:
    """
    LLM-based warm start strategy for autotuning.

    This approach uses an LLM to analyze top-performing configurations
    and generate novel candidates that exploit observed patterns.

    Key insight from EvoX: The selection mechanism (how we choose which
    configs to try) matters as much as the search strategy.
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet",
        top_k: int = 20,
        num_suggestions: int = 10,
    ):
        """
        Initialize the LLM warm start strategy.

        Args:
            model: LLM model to use
            top_k: Number of top configs to analyze
            num_suggestions: Number of novel configs to generate
        """
        self.model = model
        self.top_k = top_k
        self.num_suggestions = num_suggestions

    def create_analysis_prompt(
        self,
        configs: List[Dict[str, Any]],
        latencies: List[float],
        config_space_description: str,
    ) -> str:
        """
        Create a prompt for the LLM to analyze configurations.

        Args:
            configs: Top-k configurations
            latencies: Corresponding latencies
            config_space_description: Description of the config space

        Returns:
            Prompt string for the LLM
        """
        # Sort by latency
        sorted_pairs = sorted(zip(configs, latencies), key=lambda x: x[1])
        top_configs = sorted_pairs[:self.top_k]

        configs_json = json.dumps([
            {"config": c, "latency_ms": l}
            for c, l in top_configs
        ], indent=2)

        prompt = f"""You are an expert in GPU kernel autotuning. Analyze these top-performing
configurations and their latencies for a Triton kernel:

## Configuration Space
{config_space_description}

## Top {len(top_configs)} Configurations (sorted by latency, best first):
```json
{configs_json}
```

## Task
Based on patterns in the top configurations:
1. Identify what parameter values correlate with good performance
2. Generate {self.num_suggestions} novel configurations that might perform even better
3. Explain your reasoning for each suggestion

## Output Format
Return a JSON array with {self.num_suggestions} objects, each containing:
- "config": the full configuration dict
- "reasoning": why this config might work well
- "confidence": your confidence level (0-1)

Focus on:
- Block sizes that balance parallelism and memory access
- Number of warps that maximize occupancy
- Pipeline stages that hide latency

Generate configs that are DIFFERENT from the top performers but exploit observed patterns."""

        return prompt

    def parse_llm_response(self, response: str) -> List[LLMSuggestion]:
        """
        Parse LLM response into configuration suggestions.

        Args:
            response: LLM response text

        Returns:
            List of LLMSuggestion objects
        """
        suggestions = []

        try:
            # Try to extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                for item in data:
                    suggestions.append(LLMSuggestion(
                        config=item.get("config", {}),
                        reasoning=item.get("reasoning", ""),
                        confidence=item.get("confidence", 0.5),
                    ))
        except json.JSONDecodeError:
            # Fallback: try to extract configs line by line
            pass

        return suggestions

    def generate_seed_configs(
        self,
        evaluated_configs: List[Dict[str, Any]],
        latencies: List[float],
        config_space_description: str,
        llm_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate seed configurations using LLM analysis.

        Args:
            evaluated_configs: All evaluated configurations
            latencies: Corresponding latencies
            config_space_description: Description of config space
            llm_client: Optional LLM client (uses mock if not provided)

        Returns:
            List of novel configurations to try
        """
        # Create prompt
        prompt = self.create_analysis_prompt(
            evaluated_configs,
            latencies,
            config_space_description,
        )

        if llm_client is not None:
            # Use real LLM
            response = llm_client.complete(prompt)
        else:
            # Mock response for demo
            response = self._mock_llm_response(evaluated_configs, latencies)

        # Parse suggestions
        suggestions = self.parse_llm_response(response)

        # Sort by confidence and return configs
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return [s.config for s in suggestions]

    def _mock_llm_response(
        self,
        configs: List[Dict[str, Any]],
        latencies: List[float],
    ) -> str:
        """
        Generate a mock LLM response for demo purposes.

        This simulates what an LLM might suggest based on the top configs.
        """
        # Analyze top configs to find patterns
        sorted_pairs = sorted(zip(configs, latencies), key=lambda x: x[1])
        top_configs = [c for c, _ in sorted_pairs[:10]]

        # Find common values in top configs
        param_counts: Dict[str, Dict[Any, int]] = {}
        for config in top_configs:
            for key, value in config.items():
                if key not in param_counts:
                    param_counts[key] = {}
                if value not in param_counts[key]:
                    param_counts[key][value] = 0
                param_counts[key][value] += 1

        # Generate suggestions based on most common values
        suggestions = []
        for i in range(self.num_suggestions):
            config = {}
            for key, counts in param_counts.items():
                # Pick most common value with some variation
                sorted_values = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                idx = min(i % len(sorted_values), len(sorted_values) - 1)
                config[key] = sorted_values[idx][0]

            suggestions.append({
                "config": config,
                "reasoning": f"Based on pattern analysis of top {len(top_configs)} configs",
                "confidence": 0.7 - (i * 0.05),
            })

        return json.dumps(suggestions, indent=2)


def apply_llm_warm_start(
    search_instance: Any,
    evaluated_configs: List[Dict[str, Any]],
    latencies: List[float],
    config_space_description: str = "Triton kernel with BLOCK_SIZE, num_warps, and num_stages parameters",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply LLM warm start to inject new seed configs into search.

    This is the main entry point for Path B fallback strategy.

    Args:
        search_instance: The autotuning search instance
        evaluated_configs: Configs evaluated so far
        latencies: Corresponding latencies
        config_space_description: Description of the config space
        verbose: Whether to print progress

    Returns:
        List of LLM-suggested configs that were injected
    """
    strategy = LLMWarmStartStrategy(
        top_k=20,
        num_suggestions=10,
    )

    if verbose:
        print("\n[LLM Warm Start] Analyzing top configurations...")

    # Generate seed configs
    seed_configs = strategy.generate_seed_configs(
        evaluated_configs=evaluated_configs,
        latencies=latencies,
        config_space_description=config_space_description,
    )

    if verbose:
        print(f"[LLM Warm Start] Generated {len(seed_configs)} novel configurations")
        for i, config in enumerate(seed_configs[:3]):
            print(f"  {i+1}. {config}")
        if len(seed_configs) > 3:
            print(f"  ... and {len(seed_configs) - 3} more")

    return seed_configs
