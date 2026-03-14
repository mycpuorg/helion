"""
Rotary Position Embedding (RoPE) kernel with configurable autotuning parameters.
Essential for position encoding in modern transformers like LLaMA.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import triton
import triton.language as tl
import math


@dataclass
class RoPEConfig:
    """Configuration space for RoPE kernel autotuning."""
    BLOCK_SIZE: int = 1024
    num_warps: int = 4
    num_stages: int = 2

    BLOCK_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    NUM_WARPS_OPTIONS: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    NUM_STAGES_OPTIONS: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    @property
    def config_space_size(self) -> int:
        return (
            len(self.BLOCK_SIZE_OPTIONS) *
            len(self.NUM_WARPS_OPTIONS) *
            len(self.NUM_STAGES_OPTIONS)
        )

    def to_dict(self) -> dict:
        return {
            "BLOCK_SIZE": self.BLOCK_SIZE,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoPEConfig":
        return cls(
            BLOCK_SIZE=d.get("BLOCK_SIZE", 1024),
            num_warps=d.get("num_warps", 4),
            num_stages=d.get("num_stages", 2),
        )

    def encode(self) -> Tuple[int, ...]:
        return (
            self.BLOCK_SIZE_OPTIONS.index(self.BLOCK_SIZE) if self.BLOCK_SIZE in self.BLOCK_SIZE_OPTIONS else 0,
            self.NUM_WARPS_OPTIONS.index(self.num_warps) if self.num_warps in self.NUM_WARPS_OPTIONS else 0,
            self.NUM_STAGES_OPTIONS.index(self.num_stages) if self.num_stages in self.NUM_STAGES_OPTIONS else 0,
        )


@triton.jit
def _rope_kernel(
    # Inputs
    x_ptr,
    cos_ptr,
    sin_ptr,
    # Output
    out_ptr,
    # Dimensions
    seq_len,
    head_dim,
    # Strides
    stride_x_seq,
    stride_x_dim,
    stride_out_seq,
    stride_out_dim,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embedding."""
    pid_seq = tl.program_id(0)
    pid_dim_block = tl.program_id(1)

    # Each thread block handles BLOCK_SIZE elements of head_dim
    dim_start = pid_dim_block * BLOCK_SIZE
    dim_offs = dim_start + tl.arange(0, BLOCK_SIZE)

    # Only process half of head_dim (pairs)
    half_dim = head_dim // 2
    mask = dim_offs < half_dim

    # Load x values (real and imaginary pairs)
    x_real_ptr = x_ptr + pid_seq * stride_x_seq + dim_offs * stride_x_dim
    x_imag_ptr = x_ptr + pid_seq * stride_x_seq + (dim_offs + half_dim) * stride_x_dim

    x_real = tl.load(x_real_ptr, mask=mask, other=0.0)
    x_imag = tl.load(x_imag_ptr, mask=mask, other=0.0)

    # Load cos and sin for this position
    cos = tl.load(cos_ptr + pid_seq * half_dim + dim_offs, mask=mask, other=1.0)
    sin = tl.load(sin_ptr + pid_seq * half_dim + dim_offs, mask=mask, other=0.0)

    # Apply rotation
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    # Store results
    out_real_ptr = out_ptr + pid_seq * stride_out_seq + dim_offs * stride_out_dim
    out_imag_ptr = out_ptr + pid_seq * stride_out_seq + (dim_offs + half_dim) * stride_out_dim

    tl.store(out_real_ptr, out_real, mask=mask)
    tl.store(out_imag_ptr, out_imag, mask=mask)


def rope_kernel(
    x,
    cos,
    sin,
    config: Optional[RoPEConfig] = None,
):
    """
    RoPE forward pass with configurable autotuning parameters.

    Args:
        x: Input tensor of shape (seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim // 2)
        sin: Sine values of shape (seq_len, head_dim // 2)
        config: RoPEConfig with tuning parameters

    Returns:
        Output tensor of same shape as input
    """
    import torch

    if config is None:
        config = RoPEConfig()

    seq_len, head_dim = x.shape
    half_dim = head_dim // 2

    out = torch.empty_like(x)

    grid = (seq_len, triton.cdiv(half_dim, config.BLOCK_SIZE))

    _rope_kernel[grid](
        x, cos, sin, out,
        seq_len, head_dim,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=config.BLOCK_SIZE,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return out


def get_rope_config_space() -> List[RoPEConfig]:
    """Generate all possible configurations for autotuning."""
    configs = []
    default = RoPEConfig()

    for block_size in default.BLOCK_SIZE_OPTIONS:
        for num_warps in default.NUM_WARPS_OPTIONS:
            for num_stages in default.NUM_STAGES_OPTIONS:
                configs.append(RoPEConfig(
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                ))

    return configs
