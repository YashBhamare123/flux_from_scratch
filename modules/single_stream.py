import torch
from einops import rearrange
from mlp import MLP
import torch.nn as nn
from modulations import GetModulations
from attention import MultiHeadAttention, attention, QKNorm
from rope import generate_rope


class SingleStreamBlock(nn.Module):
    def __init__(self, dim : int, heads : int, device = 'cpu', eps = 1e-6, ):
        super().__init__()
        self.modulation = GetModulations(dim, double= False)
        self.linear1 = nn.Linear(dim, 7 * dim)
        self.linear2 = nn.Linear(5 * dim, dim)
        self.norm = QKNorm(d_model= dim // heads, eps = eps)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine= False)
        self.heads = heads
        self.device = device

    def forward(self, x : torch.Tensor, vec : torch.Tensor):
        B, N, D = x.size()
        out = self.layer_norm(x)
        modulations = self.modulation(vec)
        out = out * modulations[0].scale + modulations[0].bias

        out = rearrange(self.linear1(out), pattern= 'B N (X D) -> X B N D', X = 7)
        qkv, mlp = out[:3], out[3:]

        q, k, v = rearrange(qkv, pattern= 'X B N (Y D) -> X B Y N D', Y = self.heads)
        q, k = self.norm(q, k)
        pos = torch.arange(0, N, 1, dtype = torch.float16, device= self.device).unsqueeze(0).repeat(B, self.heads, 1)
        rot_tensor = generate_rope(pos, dim = D // self.heads, base = 10000)
        out_attention = attention(q, k, v, rot_tensor)

        mlp = rearrange(mlp, pattern= 'X B N D -> B N (X D)')
        out_mlp = nn.GELU(approximate= 'tanh')(mlp)
        out = self.linear2(torch.concat([out_attention, out_mlp], dim = -1))
        out = modulations[0].gate * out
        return x + out


if __name__ == "__main__":
    size = [64, 128, 3072]
    inp = torch.randn(size)
    vec = torch.randn(size)
    model = SingleStreamBlock(3072, 4)
    out = model(inp, vec)
    print(out.size())








