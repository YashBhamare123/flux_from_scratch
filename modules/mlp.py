import torch
import torch.nn as nn
from torch import Tensor

class Expert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_1 = nn.Parameter(Tensor([in_dim, hidden_dim]), requires_grad= True)
        self.bias_1 = nn.Parameter(Tensor([hidden_dim]))
        self.relu = nn.ReLU()
        self.layer_2 = nn.Parameter(Tensor([hidden_dim, in_dim]), requires_grad= True)
        self.bias_2 = nn.Parameter(Tensor([in_dim]))

    def forward(self):
        raise NotImplementedError

# TODO test the MOE MLP
class MLPMoE(nn.Module):
    def __init__(self, in_dim :int, hidden_dim: int, num_experts: int, k : int):
        super().__init__()
        self.gate = nn.Linear(in_dim, num_experts)
        self.layers = nn.ModuleList([Expert(in_dim, hidden_dim) for _ in range(num_experts)])
        self.k = k
        self.num_experts = num_experts

    def forward(self, x):
        # TODO Benchmark if doing all MOE comps is faster than creating an MOE weight tensor at runtime
        logits = self.gate(x)
        probs =nn.functional.softmax(logits, dim = -1)
        weights, indices = torch.topk(probs, dim = -1, k = self.k)
        weights_l1 = torch.stack([self.layers[i].layer_1 for i in indices])
        weights_l2 = torch.stack([self.layers[i].layer_2 for i in indices])
        bias_l1 =  torch.stack([self.layers[i].bias_1 for i in indices])
        bias_l2 =  torch.stack([self.layers[i].bias_2 for i in indices])
        out = x @ weights_l1 + bias_l1
        out = nn.GELU(approximate= 'tanh')(out)
        out = out @ weights_l2 + bias_l2
        out = torch.einsum('b n d, b -> n d', out, weights)
        return out



class MLP(nn.Module):
    def __init__(self, in_dim : int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(approximate= 'tanh'),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x : Tensor):
        out = self.layers(x)
        return out

if __name__ == "__main__":
    x = torch.randn([64, 4, 128, 3072])
    mlp = MLP(3072, 4 * 3072)
    print(mlp(x).size())
