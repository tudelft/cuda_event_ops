from math import floor

import torch

from iterative_3d_warp import iterative_3d_warp
from test_warp import iterative_3d_warp_torch


def trilinear_splat_torch(events, grid_resolution):
    """
    Trilinearly splats events into a grid.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 4), where each event has (x, y, z, value).
        grid_resolution (tuple): The resolution of the output grid (d, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, d, h, w) with the splatted values.
    """
    b, n, _ = events.shape
    d, h, w = grid_resolution
    output = torch.zeros(b, d, h, w, dtype=events.dtype, device=events.device)

    for batch_idx in range(b):
        for event_idx in range(n):
            x, y, z, value = events[batch_idx, event_idx]

            # determine voxel indices
            x0, y0, z0 = floor(x), floor(y), floor(z)
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

            # compute weights
            xd, yd, zd = x - x0, y - y0, z - z0
            wx0, wy0, wz0 = 1 - xd, 1 - yd, 1 - zd
            wx1, wy1, wz1 = xd, yd, zd

            # make sure indices are within bounds
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= z0 < d:
                output[batch_idx, z0, y0, x0] += value * wx0 * wy0 * wz0
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z0 < d:
                output[batch_idx, z0, y1, x0] += value * wx1 * wy0 * wz0
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[batch_idx, z0, y0, x1] += value * wx0 * wy1 * wz0
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[batch_idx, z1, y0, x0] += value * wx0 * wy0 * wz1
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[batch_idx, z0, y1, x1] += value * wx1 * wy1 * wz0
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[batch_idx, z1, y0, x1] += value * wx0 * wy1 * wz1
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[batch_idx, z1, y1, x0] += value * wx1 * wy0 * wz1
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[batch_idx, z1, y1, x1] += value * wx1 * wy1 * wz1

    return output


if __name__ == "__main__":
    # test 1: deterministic xyz and flow
    flow_mag = 0.5
    b, d, h, w = 1, 3, 5, 5
    events = torch.tensor([[[1, 1, 1.5, 0.9]]], device="cuda")  # (b, n, 4): x, y, z, val
    flows = torch.ones(b, d, h, w, 2, device="cuda") * flow_mag  # (b, d, h, w, 2): u, v flow from z to z+1

    flows_cuda = flows.clone().requires_grad_()
    flows_torch = flows.clone().requires_grad_()
    warped_events_cuda = iterative_3d_warp(events, flows_cuda)  # (b, n, d + 1, 5): x, y, z, z_orig, val
    warped_events_torch = iterative_3d_warp_torch(events, flows_torch)
    print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
    print(f"Warped events (cuda) with shape {tuple(warped_events_cuda.shape)}:\n{warped_events_cuda}\n")
    print(f"Warped events (torch) with shape {tuple(warped_events_torch.shape)}:\n{warped_events_torch}\n")

    warped_events_cuda = torch.cat([warped_events_cuda[:, :, :, :3], warped_events_cuda[:, :, :, 4:]], dim=-1)  # (b, n, d + 1, 4): x, y, z, val
    warped_events_torch = torch.cat([warped_events_torch[:, :, :, :3], warped_events_torch[:, :, :, 4:]], dim=-1)
    splatted_cuda = trilinear_splat_torch(warped_events_cuda.view(b, -1, 4), (d + 1, h, w))
    splatted_torch = trilinear_splat_torch(warped_events_torch.view(b, -1, 4), (d + 1, h, w))
    loss_cuda = splatted_cuda.diff(dim=1).abs().sum()
    loss_torch = splatted_torch.diff(dim=1).abs().sum()
    loss_cuda.backward()
    loss_torch.backward()
    print(f"Splatted image (cuda) with shape {tuple(splatted_cuda.shape)}:\n{splatted_cuda}\n")
    print(f"Splatted image (torch) with shape {tuple(splatted_torch.shape)}:\n{splatted_torch}\n")
    print(f"Flow gradients (cuda) :\n{flows_cuda.grad}\n")
    print(f"Flow gradients (torch) :\n{flows_torch.grad}\n")
