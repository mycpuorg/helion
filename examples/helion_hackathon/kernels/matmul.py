"""
Matrix multiplication kernel with configurable autotuning parameters.
The largest configuration space among common kernels.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import triton
import triton.language as tl


@dataclass
class MatmulConfig:
    """Configuration space for Matmul kernel autotuning."""
    BLOCK_SIZE_M: int = 128
    BLOCK_SIZE_N: int = 128
    BLOCK_SIZE_K: int = 32
    GROUP_SIZE_M: int = 8
    num_warps: int = 8
    num_stages: int = 3

    BLOCK_SIZE_M_OPTIONS: List[int] = field(default_factory=lambda: [64, 128, 256])
    BLOCK_SIZE_N_OPTIONS: List[int] = field(default_factory=lambda: [64, 128, 256])
    BLOCK_SIZE_K_OPTIONS: List[int] = field(default_factory=lambda: [16, 32, 64])
    GROUP_SIZE_M_OPTIONS: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    NUM_WARPS_OPTIONS: List[int] = field(default_factory=lambda: [2, 4, 8])
    NUM_STAGES_OPTIONS: List[int] = field(default_factory=lambda: [2, 3, 4, 5])

    @property
    def config_space_size(self) -> int:
        return (
            len(self.BLOCK_SIZE_M_OPTIONS) *
            len(self.BLOCK_SIZE_N_OPTIONS) *
            len(self.BLOCK_SIZE_K_OPTIONS) *
            len(self.GROUP_SIZE_M_OPTIONS) *
            len(self.NUM_WARPS_OPTIONS) *
            len(self.NUM_STAGES_OPTIONS)
        )

    def to_dict(self) -> dict:
        return {
            "BLOCK_SIZE_M": self.BLOCK_SIZE_M,
            "BLOCK_SIZE_N": self.BLOCK_SIZE_N,
            "BLOCK_SIZE_K": self.BLOCK_SIZE_K,
            "GROUP_SIZE_M": self.GROUP_SIZE_M,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatmulConfig":
        return cls(
            BLOCK_SIZE_M=d.get("BLOCK_SIZE_M", 128),
            BLOCK_SIZE_N=d.get("BLOCK_SIZE_N", 128),
            BLOCK_SIZE_K=d.get("BLOCK_SIZE_K", 32),
            GROUP_SIZE_M=d.get("GROUP_SIZE_M", 8),
            num_warps=d.get("num_warps", 8),
            num_stages=d.get("num_stages", 3),
        )

    def encode(self) -> Tuple[int, ...]:
        return (
            self.BLOCK_SIZE_M_OPTIONS.index(self.BLOCK_SIZE_M) if self.BLOCK_SIZE_M in self.BLOCK_SIZE_M_OPTIONS else 0,
            self.BLOCK_SIZE_N_OPTIONS.index(self.BLOCK_SIZE_N) if self.BLOCK_SIZE_N in self.BLOCK_SIZE_N_OPTIONS else 0,
            self.BLOCK_SIZE_K_OPTIONS.index(self.BLOCK_SIZE_K) if self.BLOCK_SIZE_K in self.BLOCK_SIZE_K_OPTIONS else 0,
            self.GROUP_SIZE_M_OPTIONS.index(self.GROUP_SIZE_M) if self.GROUP_SIZE_M in self.GROUP_SIZE_M_OPTIONS else 0,
            self.NUM_WARPS_OPTIONS.index(self.num_warps) if self.num_warps in self.NUM_WARPS_OPTIONS else 0,
            self.NUM_STAGES_OPTIONS.index(self.num_stages) if self.num_stages in self.NUM_STAGES_OPTIONS else 0,
        )


@triton.jit
def _matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (constexpr for compilation)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """High-performance matrix multiplication kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_kernel(
    a,
    b,
    config: Optional[MatmulConfig] = None,
):
    """
    Matrix multiplication with configurable autotuning parameters.

    Args:
        a: Input tensor A of shape (M, K)
        b: Input tensor B of shape (K, N)
        config: MatmulConfig with tuning parameters

    Returns:
        Output tensor of shape (M, N)
    """
    import torch

    if config is None:
        config = MatmulConfig()

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Matrices must be contiguous"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=config.BLOCK_SIZE_M,
        BLOCK_SIZE_N=config.BLOCK_SIZE_N,
        BLOCK_SIZE_K=config.BLOCK_SIZE_K,
        GROUP_SIZE_M=config.GROUP_SIZE_M,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return c


def get_matmul_config_space() -> List[MatmulConfig]:
    """Generate all possible configurations for autotuning."""
    configs = []
    default = MatmulConfig()

    for block_m in default.BLOCK_SIZE_M_OPTIONS:
        for block_n in default.BLOCK_SIZE_N_OPTIONS:
            for block_k in default.BLOCK_SIZE_K_OPTIONS:
                for group_m in default.GROUP_SIZE_M_OPTIONS:
                    for num_warps in default.NUM_WARPS_OPTIONS:
                        for num_stages in default.NUM_STAGES_OPTIONS:
                            configs.append(MatmulConfig(
                                BLOCK_SIZE_M=block_m,
                                BLOCK_SIZE_N=block_n,
                                BLOCK_SIZE_K=block_k,
                                GROUP_SIZE_M=group_m,
                                num_warps=num_warps,
                                num_stages=num_stages,
                            ))

    return configs
