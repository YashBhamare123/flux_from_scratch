import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_func
except Exception as _:
    flash_attn = None
    flash_attn_func = None
    fl_attn = False

from rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, eps: float, d_model: int):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(d_model))
    def norm(self, x):
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        out = (x / rms) * self.scale
        return out

    def forward(self, q, k):
        raise NotImplementedError


class QKNorm(RMSNorm):
    def __init__(self, eps: float, d_model: int):
        super().__init__(eps, d_model)
        self.key_norm = RMSNorm(self.eps, self.d_model)
        self.query_norm = RMSNorm(self.eps, self.d_model)

    def forward(self, q, k):
        k = self.key_norm.norm(k)
        q = self.query_norm.norm(q)
        return q, k


def attention(q : Tensor, k: Tensor, v: Tensor, pe, attn = 'sdpa'):
    k = apply_rope(k, pe)
    q = apply_rope(q, pe)

    if attn == 'sdpa':
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'B H L D -> B L (H D)')
        return out

    if attn == 'naive':
        dim = torch.tensor(v.size()[-1])
        out = q @ k.permute(0, 1, 3, 2)
        scale = F.softmax(out, dim = 1)/ torch.sqrt(dim)
        out = scale @ v
        return out

    if attn == 'flash':
        if not fl_attn:
            raise ImportError('Flash Attention is not available. Please install')
        else:
            out = flash_attn_func(q, k, v)
            out = rearrange(out, 'B H L D -> B L (HD)')


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, eps : float, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = QKNorm(eps, int(dim/heads))

    def forward(self, x: Tensor, pe: Tensor):
        raise NotImplementedError


if __name__ == '__main__':
    shape = [64, 8, 128, 768]
    eps = 1e-6


