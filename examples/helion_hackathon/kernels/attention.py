"""
Flash Attention kernel with configurable autotuning parameters.
Critical for transformer model performance.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import triton
import triton.language as tl
import math


@dataclass
class AttentionConfig:
    """Configuration space for Attention kernel autotuning."""
    BLOCK_M: int = 128
    BLOCK_N: int = 64
    BLOCK_DMODEL: int = 64
    num_warps: int = 4
    num_stages: int = 2

    BLOCK_M_OPTIONS: List[int] = field(default_factory=lambda: [64, 128, 256])
    BLOCK_N_OPTIONS: List[int] = field(default_factory=lambda: [32, 64, 128])
    BLOCK_DMODEL_OPTIONS: List[int] = field(default_factory=lambda: [32, 64, 128])
    NUM_WARPS_OPTIONS: List[int] = field(default_factory=lambda: [2, 4, 8])
    NUM_STAGES_OPTIONS: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    @property
    def config_space_size(self) -> int:
        return (
            len(self.BLOCK_M_OPTIONS) *
            len(self.BLOCK_N_OPTIONS) *
            len(self.BLOCK_DMODEL_OPTIONS) *
            len(self.NUM_WARPS_OPTIONS) *
            len(self.NUM_STAGES_OPTIONS)
        )

    def to_dict(self) -> dict:
        return {
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_DMODEL": self.BLOCK_DMODEL,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttentionConfig":
        return cls(
            BLOCK_M=d.get("BLOCK_M", 128),
            BLOCK_N=d.get("BLOCK_N", 64),
            BLOCK_DMODEL=d.get("BLOCK_DMODEL", 64),
            num_warps=d.get("num_warps", 4),
            num_stages=d.get("num_stages", 2),
        )

    def encode(self) -> Tuple[int, ...]:
        return (
            self.BLOCK_M_OPTIONS.index(self.BLOCK_M) if self.BLOCK_M in self.BLOCK_M_OPTIONS else 0,
            self.BLOCK_N_OPTIONS.index(self.BLOCK_N) if self.BLOCK_N in self.BLOCK_N_OPTIONS else 0,
            self.BLOCK_DMODEL_OPTIONS.index(self.BLOCK_DMODEL) if self.BLOCK_DMODEL in self.BLOCK_DMODEL_OPTIONS else 0,
            self.NUM_WARPS_OPTIONS.index(self.num_warps) if self.num_warps in self.NUM_WARPS_OPTIONS else 0,
            self.NUM_STAGES_OPTIONS.index(self.num_stages) if self.num_stages in self.NUM_STAGES_OPTIONS else 0,
        )


@triton.jit
def _attention_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention forward kernel."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers to Q, K, V
    off_q = off_z * stride_qz + off_h * stride_qh
    off_k = off_z * stride_kz + off_h * stride_kh
    off_v = off_z * stride_vz + off_h * stride_vh

    q_ptrs = Q + off_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_k + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + off_v + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q block (stays in SRAM)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Compute softmax
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update running max and sum
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Rescale previous accumulator
        p_scale = beta / l_i_new
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        # Load V block and update accumulator
        v = tl.load(v_ptrs + start_n * stride_vk,
                    mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        p = p * p_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)

        # Update m_i and l_i
        m_i = m_i_new
        l_i = l_i_new

    # Write output
    off_o = off_z * stride_oz + off_h * stride_oh
    out_ptrs = Out + off_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def attention_kernel(
    q, k, v,
    config: Optional[AttentionConfig] = None,
):
    """
    Flash attention forward pass with configurable autotuning parameters.

    Args:
        q: Query tensor of shape (Z, H, N_CTX, D_MODEL)
        k: Key tensor of shape (Z, H, N_CTX, D_MODEL)
        v: Value tensor of shape (Z, H, N_CTX, D_MODEL)
        config: AttentionConfig with tuning parameters

    Returns:
        Output tensor of shape (Z, H, N_CTX, D_MODEL)
    """
    import torch

    if config is None:
        config = AttentionConfig()

    Z, H, N_CTX, D_MODEL = q.shape

    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D_MODEL)

    grid = (triton.cdiv(N_CTX, config.BLOCK_M), Z * H)

    _attention_kernel[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=config.BLOCK_M,
        BLOCK_DMODEL=config.BLOCK_DMODEL,
        BLOCK_N=config.BLOCK_N,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return o


def get_attention_config_space() -> List[AttentionConfig]:
    """Generate all possible configurations for autotuning."""
    configs = []
    default = AttentionConfig()

    for block_m in default.BLOCK_M_OPTIONS:
        for block_n in default.BLOCK_N_OPTIONS:
            for block_d in default.BLOCK_DMODEL_OPTIONS:
                for num_warps in default.NUM_WARPS_OPTIONS:
                    for num_stages in default.NUM_STAGES_OPTIONS:
                        configs.append(AttentionConfig(
                            BLOCK_M=block_m,
                            BLOCK_N=block_n,
                            BLOCK_DMODEL=block_d,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        ))

    return configs
