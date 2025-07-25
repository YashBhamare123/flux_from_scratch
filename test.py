from einops import rearrange
import torch

inp = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
out = rearrange(inp, pattern= '(X H Y) -> X H Y', X = 3, H = 3)
print(out.size())
print(out)