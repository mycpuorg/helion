import math
from dataclasses import dataclass

import helion
import helion.language as hl
import torch
import torch.nn as nn


@dataclass
class Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    helion: bool = True


@helion.kernel(
    # Static shapes provides a speedup for attention
    static_shapes=True
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Implements the attention mechanism: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim]
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.config.helion:
            y = attention(q, k, v)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C) where C == config.n_embd

        Returns:
            Tensor of shape (B, T, C)
        """
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


def test_transformer():
    torch.manual_seed(0)

    device = torch.device("cuda")

    config = Config()
    model = Transformer(config).to(device)

    B, T = 2, 16
    x = torch.randn(B, T, config.n_embd, device=device)

    y = model(x)
    print("output shape:", y.shape)
    assert y.shape == x.shape

    loss = y.pow(2).mean()
    loss.backward()
    print("backward OK, loss:", loss.item())


def test_helion_parity():
    torch.manual_seed(0)

    device = torch.device("cuda")

    helion_model = Transformer(Config(helion=True)).to(device)
    torch_model = Transformer(Config(helion=False)).to(device)
    torch_model.load_state_dict(helion_model.state_dict())

    B, T = 2, 16
    x_helion = torch.randn(B, T, helion_model.ln_f.normalized_shape[0], device=device)
    x_helion.requires_grad_()
    x_torch = x_helion.clone().detach().requires_grad_(True)

    y_helion = helion_model(x_helion)
    y_torch = torch_model(x_torch)
    torch.testing.assert_close(y_helion, y_torch, rtol=1e-3, atol=1e-3)

    print("||y_helion||:", y_helion.norm().item())
    print("||y_torch||:", y_torch.norm().item())
    
    helion_model.zero_grad()
    torch_model.zero_grad()
    loss_helion = y_helion.pow(2).mean()
    loss_torch = y_torch.pow(2).mean()
    loss_helion.backward()
    loss_torch.backward()

    torch.testing.assert_close(x_helion.grad, x_torch.grad, rtol=1e-3, atol=1e-3)
    for (name_h, param_h), (name_t, param_t) in zip(
        helion_model.named_parameters(), torch_model.named_parameters()
    ):
        torch.testing.assert_close(
            param_h.grad,
            param_t.grad,
            rtol=1e-3,
            atol=1e-3,
            msg=f"Gradient mismatch for {name_h} vs {name_t}",
        )



# test_transformer()
test_helion_parity()
