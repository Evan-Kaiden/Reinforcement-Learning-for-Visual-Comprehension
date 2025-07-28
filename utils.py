import torch

def make_glimpse_grid(center, patch_size, image_size):
    """
    center: tensor of shape [B, 2], in [-1, 1] (normalized x, y)
    patch_size: int (e.g., 12)
    image_size: int (assume square, e.g., 64)
    Returns:
        grid: [B, patch_size, patch_size, 2] for grid_sample
    """
    B = center.size(0)
    g = patch_size

    lin = torch.linspace(-1, 1, g, device=center.device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')  # [g, g]
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [g, g, 2]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, g, g, 2]

    # Add center offset
    center = center.view(B, 1, 1, 2)  # [B, 1, 1, 2]
    grid = base_grid * (patch_size / image_size) + center  # scale base_grid before adding

    return grid.clamp(-1, 1)
