import torch

def entropy_weight_t(n, total, start_entropy):
    """scale linearly from start_entropy -> 0 over T timesteps"""
    return start_entropy * (1 - n / total)

def map_scale(val, img_size):
    """Maps from range [-1, 1] to range [0, img_size]"""
    val = ((val + 1) / 2) * img_size
    return val

def make_glimpse_grid(center, p, S, align_corners=True):
    """Returns a grid [B,p,p,2] a sampling grid that maps [-1,1] to a location"""
    B = center.size(0)

    # Create Grid
    lin = torch.linspace(-1, 1, p, device=center.device, dtype=center.dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Create Value Scale
    if align_corners:
        scale = (p - 1) / max(S - 1, 1)
    else:
        scale = p / S

    # Scale Grid
    grid = center.view(B, 1, 1, 2) + scale * base
    return grid.clamp_(-1, 1)