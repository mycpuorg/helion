"""
LayerNorm kernel with configurable autotuning parameters.
This kernel is a prime target for autotuning due to its high wall-time
contribution in transformer workloads and large configuration space.
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import triton
import triton.language as tl


@dataclass
class LayerNormConfig:
    """Configuration space for LayerNorm kernel autotuning."""
    BLOCK_SIZE: int = 1024
    num_warps: int = 8
    num_stages: int = 2
    eps: float = 1e-5

    # Configuration space bounds for autotuning
    BLOCK_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
    NUM_WARPS_OPTIONS: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    NUM_STAGES_OPTIONS: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    @property
    def config_space_size(self) -> int:
        """Total number of possible configurations."""
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
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerNormConfig":
        return cls(
            BLOCK_SIZE=d.get("BLOCK_SIZE", 1024),
            num_warps=d.get("num_warps", 8),
            num_stages=d.get("num_stages", 2),
            eps=d.get("eps", 1e-5),
        )

    def encode(self) -> Tuple[int, ...]:
        """Encode config as tuple for diversity calculations."""
        return (
            self.BLOCK_SIZE_OPTIONS.index(self.BLOCK_SIZE) if self.BLOCK_SIZE in self.BLOCK_SIZE_OPTIONS else 0,
            self.NUM_WARPS_OPTIONS.index(self.num_warps) if self.num_warps in self.NUM_WARPS_OPTIONS else 0,
            self.NUM_STAGES_OPTIONS.index(self.num_stages) if self.num_stages in self.NUM_STAGES_OPTIONS else 0,
        )


@triton.jit
def _layernorm_fwd_kernel(
    X,  # input tensor
    Y,  # output tensor
    W,  # weight (gamma)
    B,  # bias (beta)
    Mean,  # mean output
    Rstd,  # reciprocal std output
    stride_x,  # row stride
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """Forward pass of LayerNorm kernel."""
    row = tl.program_id(0)

    # Compute row pointer
    X += row * stride_x
    Y += row * stride_x

    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_centered = tl.where(mask, x - mean, 0.0)
        _var += x_centered * x_centered
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply affine transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)


def layernorm_kernel(
    x,
    weight,
    bias,
    config: Optional[LayerNormConfig] = None,
):
    """
    LayerNorm forward pass with configurable autotuning parameters.

    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,)
        bias: Bias tensor of shape (N,)
        config: LayerNormConfig with tuning parameters

    Returns:
        Tuple of (output, mean, rstd)
    """
    import torch

    if config is None:
        config = LayerNormConfig()

    M, N = x.shape

    # Allocate outputs
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    # Launch kernel
    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x, y, weight, bias, mean, rstd,
        x.stride(0), N, config.eps,
        BLOCK_SIZE=config.BLOCK_SIZE,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return y, mean, rstd


def get_layernorm_config_space() -> List[LayerNormConfig]:
    """Generate all possible configurations for autotuning."""
    configs = []
    default = LayerNormConfig()

    for block_size in default.BLOCK_SIZE_OPTIONS:
        for num_warps in default.NUM_WARPS_OPTIONS:
            for num_stages in default.NUM_STAGES_OPTIONS:
                configs.append(LayerNormConfig(
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                ))

    return configs
