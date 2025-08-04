import torch


def map_scale(val, img_size):
    val = ((val + 1) / 2) * img_size
    return val

def make_glimpse_grid(center, p, S, align_corners=True):
    # center: [B,2] in [-1,1] (x,y), return [B,p,p,2]
    B = center.size(0)
    lin = torch.linspace(-1, 1, p, device=center.device, dtype=center.dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    if align_corners:
        scale = (p - 1) / max(S - 1, 1)
    else:
        scale = p / S
    grid = center.view(B, 1, 1, 2) + scale * base
    return grid.clamp_(-1, 1)