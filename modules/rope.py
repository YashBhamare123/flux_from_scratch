import torch
import torch.nn as nn
from torch import Tensor
import time

def generate_rope(pos: Tensor, dim : int, base: int) -> Tensor:
    exponent = torch.arange(0, dim, 2, device = pos.device, dtype = pos.dtype)
    scale = 1.0 / (base**exponent)
    angles = torch.einsum("...n, ...d -> ...nd", pos, scale)
    print(angles.size())
    output = torch.stack((torch.cos(angles), torch.sin(angles), - torch.sin(angles), torch.cos(angles)), dim = -1)
    output = output.contiguous().view([*output.size()[:-1], 2, 2])
    return output


def apply_rope(x: Tensor, rot_tensor: Tensor):
    shape = x.size()
    x = x.contiguous().view([*shape[:-1], -1, 1, 2])
    x = rot_tensor[..., 0] * x[... , 0] + rot_tensor[..., 1] * x[..., 1]
    x = x.reshape(shape)
    return x

if __name__ == '__main__':
    B, H, N, D = (16, 4, 128, 768)
    tensor = torch.randn([B, H, N, D])
    pos = torch.arange(0, 2 * N, 1, dtype = torch.float16, device = 'cpu').unsqueeze(0).repeat(B ,H , 1)
    rot_tensor = generate_rope(pos, D, 10000)
    rop = apply_rope(tensor, rot_tensor)
    print(rop.size())




