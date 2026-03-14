"""
Sample Transformer Workload for profiling.

Provides a realistic workload that exercises common transformer kernels:
- LayerNorm
- Softmax
- MatMul
- Attention
- RoPE
"""
import sys
import os
from typing import Optional, Dict, Callable
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class WorkloadConfig:
    """Configuration for the sample transformer workload."""
    batch_size: int = 4
    seq_len: int = 2048
    hidden_size: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 2  # Reduced for profiling


class TransformerWorkload:
    """
    A sample transformer workload that mimics inference patterns.

    This workload exercises the key kernels we want to profile:
    - LayerNorm: Pre/post normalization
    - Softmax: Attention score normalization
    - MatMul: QKV projections, FFN layers
    - Attention: Core attention computation
    - RoPE: Position embeddings
    """

    def __init__(self, config: Optional[WorkloadConfig] = None, device: str = "cuda"):
        """
        Initialize the workload.

        Args:
            config: Workload configuration
            device: Device to run on ("cuda" or "cpu")
        """
        self.config = config or WorkloadConfig()
        self.device = device
        self._initialized = False
        self._tensors = {}
        self._kernel_timings: Dict[str, float] = {}

    def setup(self):
        """Allocate tensors for the workload."""
        import torch

        cfg = self.config
        device = self.device

        # Input embeddings
        self._tensors["hidden_states"] = torch.randn(
            cfg.batch_size, cfg.seq_len, cfg.hidden_size,
            device=device, dtype=torch.float16
        )

        # LayerNorm parameters
        self._tensors["ln_weight"] = torch.ones(cfg.hidden_size, device=device, dtype=torch.float16)
        self._tensors["ln_bias"] = torch.zeros(cfg.hidden_size, device=device, dtype=torch.float16)

        # Attention inputs (for flash attention)
        self._tensors["q"] = torch.randn(
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
            device=device, dtype=torch.float16
        )
        self._tensors["k"] = torch.randn(
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
            device=device, dtype=torch.float16
        )
        self._tensors["v"] = torch.randn(
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
            device=device, dtype=torch.float16
        )

        # Softmax input
        self._tensors["attn_scores"] = torch.randn(
            cfg.batch_size * cfg.num_heads, cfg.seq_len, cfg.seq_len,
            device=device, dtype=torch.float16
        )

        # MatMul inputs
        self._tensors["mat_a"] = torch.randn(
            cfg.batch_size * cfg.seq_len, cfg.hidden_size,
            device=device, dtype=torch.float16
        )
        self._tensors["mat_b"] = torch.randn(
            cfg.hidden_size, cfg.hidden_size,
            device=device, dtype=torch.float16
        )

        # RoPE inputs
        self._tensors["rope_x"] = torch.randn(
            cfg.seq_len, cfg.head_dim,
            device=device, dtype=torch.float16
        )
        half_dim = cfg.head_dim // 2
        positions = torch.arange(cfg.seq_len, device=device, dtype=torch.float32)
        freqs = torch.exp(
            -torch.arange(0, half_dim, device=device, dtype=torch.float32)
            * (2.0 / half_dim) * 10.0  # Simplified frequency computation
        )
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        self._tensors["rope_cos"] = torch.cos(angles).to(torch.float16)
        self._tensors["rope_sin"] = torch.sin(angles).to(torch.float16)

        self._initialized = True

    def run_layernorm(self) -> "torch.Tensor":
        """Run LayerNorm kernel."""
        from kernels.layernorm import layernorm_kernel

        hidden = self._tensors["hidden_states"]
        weight = self._tensors["ln_weight"]
        bias = self._tensors["ln_bias"]

        # Reshape for 2D layernorm
        batch_seq = hidden.shape[0] * hidden.shape[1]
        hidden_2d = hidden.view(batch_seq, -1)

        output, _, _ = layernorm_kernel(hidden_2d, weight, bias)
        return output

    def run_softmax(self) -> "torch.Tensor":
        """Run Softmax kernel."""
        from kernels.softmax import softmax_kernel

        scores = self._tensors["attn_scores"]
        # Reshape for 2D softmax
        flat_scores = scores.view(-1, scores.shape[-1])
        return softmax_kernel(flat_scores)

    def run_matmul(self) -> "torch.Tensor":
        """Run MatMul kernel."""
        from kernels.matmul import matmul_kernel

        a = self._tensors["mat_a"]
        b = self._tensors["mat_b"]
        return matmul_kernel(a, b)

    def run_attention(self) -> "torch.Tensor":
        """Run Flash Attention kernel."""
        from kernels.attention import attention_kernel

        q = self._tensors["q"]
        k = self._tensors["k"]
        v = self._tensors["v"]
        return attention_kernel(q, k, v)

    def run_rope(self) -> "torch.Tensor":
        """Run RoPE kernel."""
        from kernels.rope import rope_kernel

        x = self._tensors["rope_x"]
        cos = self._tensors["rope_cos"]
        sin = self._tensors["rope_sin"]
        return rope_kernel(x, cos, sin)

    def run_all(self):
        """Run all kernels in sequence (simulates one transformer layer)."""
        if not self._initialized:
            self.setup()

        # Run each kernel multiple times to simulate a full layer
        for _ in range(self.config.num_layers):
            self.run_layernorm()
            self.run_matmul()  # QKV projection
            self.run_rope()
            self.run_attention()
            self.run_softmax()
            self.run_matmul()  # Output projection
            self.run_layernorm()
            self.run_matmul()  # FFN up
            self.run_matmul()  # FFN down

    def get_kernel_config_spaces(self) -> Dict[str, int]:
        """
        Get the configuration space size for each kernel.

        These are computed from the kernel config classes.
        """
        from kernels.layernorm import LayerNormConfig
        from kernels.softmax import SoftmaxConfig
        from kernels.matmul import MatmulConfig
        from kernels.attention import AttentionConfig
        from kernels.rope import RoPEConfig

        return {
            "layernorm": LayerNormConfig().config_space_size,
            "softmax": SoftmaxConfig().config_space_size,
            "matmul": MatmulConfig().config_space_size,
            "attention": AttentionConfig().config_space_size,
            "rope": RoPEConfig().config_space_size,
        }

    def get_kernel_functions(self) -> Dict[str, Callable[[], None]]:
        """Get kernel functions for profiling."""
        if not self._initialized:
            self.setup()

        return {
            "layernorm": self.run_layernorm,
            "softmax": self.run_softmax,
            "matmul": self.run_matmul,
            "attention": self.run_attention,
            "rope": self.run_rope,
        }


def create_profiling_workload(
    batch_size: int = 4,
    seq_len: int = 2048,
    hidden_size: int = 4096,
    device: str = "cuda",
) -> TransformerWorkload:
    """
    Create a transformer workload configured for profiling.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        device: Device to run on

    Returns:
        Configured TransformerWorkload instance
    """
    config = WorkloadConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=hidden_size // 128,
        head_dim=128,
        num_layers=2,
    )
    workload = TransformerWorkload(config=config, device=device)
    workload.setup()
    return workload
