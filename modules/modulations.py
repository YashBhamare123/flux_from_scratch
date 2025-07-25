import torch
import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
from mlp import MLP, MLPMoE


@dataclass
class Modulations:
    bias : Tensor
    scale : Tensor
    gate : Tensor

class GetModulations(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.dim = dim
        self.multiplier = 3 * double
        if double:
            self.lin = nn.Linear(dim, 6*dim)
        else:
            self.lin = nn.Linear(dim, 3*dim)
        self.double = double


    def forward(self, vec : Tensor):
        vec = nn.functional.silu(vec)
        out = self.lin(vec)
        out = (Modulations(*torch.chunk(out[..., :self.dim* 3], chunks = 3, dim = -1)),
               Modulations(*torch.chunk(out[..., self.dim *3 :], chunks= 3, dim = -1)) if self.double else None)
        return out


if __name__ == "__main__":
    x = torch.randn([64, 128, 3072])
    mod = GetModulations(3072,False)
    print(mod(x)[0].scale.size())
    print(mod(x)[0].bias.size())
    print(mod(x)[0].gate.size())
    print(mod(x)[1].scale.size())
    print(mod(x)[1].bias.size())
    print(mod(x)[1].gate.size())
