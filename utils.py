import torch

# -------- helpers --------
def make_glimpse_grid(center, g, S, align_corners=True):
    # center: [B,2] in [-1,1] (x,y), return [B,g,g,2]
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

def tiled_centers(S, g, stride=None, device=None, dtype=None):
    # row-major scan; returns [T,2] centers in [-1,1], align_corners=True
    s = g if stride is None else stride
    def axis_pos(S, g, s):
        p, out = 0, [0]
        last = S - g
        while p < last:
            p = min(p + s, last)
            out.append(p)
        return torch.tensor(out, device=device, dtype=torch.float32 if dtype is None else dtype)
    xs = axis_pos(S, g, s)
    ys = axis_pos(S, g, s)
    cx = xs + (g - 1)/2
    cy = ys + (g - 1)/2
    gy, gx = torch.meshgrid(cy, cx, indexing='ij')
    x_norm = 2*gx/(S-1) - 1
    y_norm = 2*gy/(S-1) - 1
    return torch.stack([x_norm.reshape(-1), y_norm.reshape(-1)], dim=-1)  # [T,2]
