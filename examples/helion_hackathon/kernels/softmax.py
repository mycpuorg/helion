"""
Softmax kernel with configurable autotuning parameters.
Critical for attention mechanisms in transformers.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import triton
import triton.language as tl


@dataclass
class SoftmaxConfig:
    """Configuration space for Softmax kernel autotuning."""
    BLOCK_SIZE: int = 1024
    num_warps: int = 4
    num_stages: int = 2

    BLOCK_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    NUM_WARPS_OPTIONS: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    NUM_STAGES_OPTIONS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

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
    def from_dict(cls, d: dict) -> "SoftmaxConfig":
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
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Numerically stable softmax kernel."""
    row_idx = tl.program_id(0)

    # Row start pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Load and compute max for numerical stability
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Initialize max with -inf
    row_max = float("-inf")

    # First pass: find max
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        row = tl.load(input_ptrs + start, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, tl.max(row, axis=0))

    # Second pass: compute exp(x - max) and sum
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        row = tl.load(input_ptrs + start, mask=mask, other=float("-inf"))
        row_exp = tl.exp(row - row_max)
        row_sum += tl.sum(tl.where(mask, row_exp, 0.0), axis=0)

    # Third pass: normalize
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        row = tl.load(input_ptrs + start, mask=mask, other=float("-inf"))
        row_exp = tl.exp(row - row_max)
        softmax_output = row_exp / row_sum
        tl.store(out_row_start_ptr + start + col_offsets, softmax_output, mask=mask)


def softmax_kernel(
    x,
    config: Optional[SoftmaxConfig] = None,
):
    """
    Softmax forward pass with configurable autotuning parameters.

    Args:
        x: Input tensor of shape (M, N)
        config: SoftmaxConfig with tuning parameters

    Returns:
        Output tensor of same shape as input
    """
    import torch

    if config is None:
        config = SoftmaxConfig()

    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M,)
    _softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=config.BLOCK_SIZE,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return y


def get_softmax_config_space() -> List[SoftmaxConfig]:
    """Generate all possible configurations for autotuning."""
    configs = []
    default = SoftmaxConfig()

    for block_size in default.BLOCK_SIZE_OPTIONS:
        for num_warps in default.NUM_WARPS_OPTIONS:
            for num_stages in default.NUM_STAGES_OPTIONS:
                configs.append(SoftmaxConfig(
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                ))

    return configs
