import torch
from torch import Tensor
from einops import rearrange

def generate_rope(ids : Tensor, dim :int, period :float = 10000.) -> Tensor:
    dim_ids = torch.arange(0, dim/2).unsqueeze(0)/dim
    angles = torch.exp(-torch.log(torch.tensor(period)) * 2 * dim_ids)
    ids = rearrange(ids, "... N D -> ... D N")
    angles = ids @ angles
    angle_matrix = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim = -1)   
    angle_matrix = rearrange(angle_matrix, "... (D W) -> ... D W", D = 2, W = 2)
    return angle_matrix

def apply_rope(x : Tensor, rot_tensor : Tensor) ->Tensor:
    D = x.size()[-1]
    x = rearrange(x, "... (D A) -> ... D A", D =D//2, A = 2)
    rot_tensor = rot_tensor.unsqueeze(0)
    print(rot_tensor.size())
    x = x * rot_tensor[..., 0] + x * rot_tensor[..., 1]
    x = rearrange(x, "... D A -> ... (D A)", D = D//2, A = 2)
    return x

if __name__ == '__main__':
    B, H, N, D = (16, 4, 128, 768)
    tensor = torch.randn([B, H, N, D])
    pos = torch.arange(0, N, 1, dtype = torch.float, device = 'cpu').unsqueeze(0)
    rot_tensor = generate_rope(pos, D)
    rop = apply_rope(tensor, rot_tensor)
    print(rop.size())




