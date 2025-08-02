import torch
 
def make_glimpse_grid(center, g, S, align_corners=True):
    # center: [B,2] in [-1,1] (x,y), return [B,p,p,2]
    B = center.size(0)
    lin = torch.linspace(-1, 1, g, device=center.device, dtype=center.dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    if align_corners:
        scale = (g - 1) / max(S - 1, 1)
    else:
        scale = g / S
    grid = center.view(B, 1, 1, 2) + scale * base
    return grid.clamp_(-1, 1)