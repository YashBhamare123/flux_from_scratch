import torch.nn as nn
import torch
from einops import rearrange
from torch import Tensor

from modulations import GetModulations
from attention import MultiHeadAttention
from mlp import MLP, MLPMoE
from modules.attention import attention
from rope import generate_rope


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, heads : int, eps = 1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.img_attn = MultiHeadAttention(self.dim, eps, heads)
        self.img_mlp = MLP(self.dim, 4*self.dim)
        self.img_mod = GetModulations(self.dim, double = True)
        self.txt_attn = MultiHeadAttention(self.dim, eps, heads)
        self.txt_mlp = MLP(self.dim, 4*self.dim)
        self.txt_mod = GetModulations(self.dim, double= True)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine= False)

    def forward(self, img: Tensor, text: Tensor, vec: Tensor):
        B, N, D = img.size()

        img_norm = self.layer_norm(img)
        text_norm = self.layer_norm(text)
        img_modulations = self.img_mod(vec)
        text_modulations = self.txt_mod(vec)

        img_norm = img_norm * img_modulations[0].scale + img_modulations[0].bias
        text_norm = text_norm * text_modulations[0].scale + text_modulations[0].bias

        q_img, k_img, v_img =rearrange(self.img_attn.qkv(img_norm), pattern = 'B N (X H Y) -> X B H N Y', X = 3, H = self.heads)
        q_text, k_text, v_text=rearrange(self.txt_attn.qkv(text_norm), pattern = 'B N (X H Y) -> X B H N Y', X = 3, H = self.heads)

        print(q_img.size())
        q_img, k_img = self.img_attn.norm(q_img, k_img)
        q_text, k_text = self.txt_attn.norm(q_text, q_text)

        print(q_img.size())
        q, k, v = (torch.concat([q_img, q_text], dim = -2), torch.concat([k_img, k_text], dim = -2), torch.concat([v_img, v_text], dim = -2 ))
        pos = torch.arange(0, 2*N, 1, dtype=torch.float16, device='cpu').unsqueeze(0).repeat(B, self.heads, 1)
        rot_tensor = generate_rope(pos, int(D / self.heads), 10000)
        out = attention(q, k, v, rot_tensor)

        img_norm, text_norm = rearrange(out, pattern= 'B (X L) D -> X B L D', X = 2)
        img_norm = self.img_attn.proj(img_norm)
        text_norm = self.txt_attn.proj(text_norm)

        img_norm = img_modulations[0].gate * img_norm
        text_norm = text_modulations[0].gate * text_norm

        img_norm = img + img_norm
        text_norm = text + text_norm

        img_norm = self.layer_norm(img_norm)
        text_norm = self.layer_norm(text_norm)

        img_norm = img_norm * img_modulations[1].scale + img_modulations[1].bias
        text_norm = text_norm * text_modulations[1].scale + text_modulations[1].bias
        img_norm = self.img_mlp(img_norm)
        text_norm = self.txt_mlp(text_norm)
        img_norm = img_modulations[1].gate * img_norm
        text_norm = text_modulations[1].gate * text_norm

        img_norm = img + img_norm
        text_norm = text + text_norm

        return text_norm, img_norm


if __name__ == "__main__":
    size = [64, 128, 3072]
    img = torch.randn(size)
    text = torch.randn(size)
    vec = torch.randn(size)

    model = DoubleStreamBlock(dim = 3072, heads = 4)
    img, text = model(img, text, vec)








