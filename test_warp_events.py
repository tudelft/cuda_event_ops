from math import floor, ceil

import torch

from cuda_3d_ops import iterative_3d_warp_cuda, trilinear_splat_cuda
from test_3d_warp import visualize_tensor


def iterative_3d_warp_torch(events, flows, mode="bilinear"):
    """
    Iteratively warps events in 3D using flow fields and bilinear/trilinear interpolation.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, zi, val).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).

    Returns:
        torch.Tensor: A tensor of shape (b, n, d + 1, 5), where each event has (x, y, z, z_orig, val).
    """
    b, n, _ = events.shape
    _, d, h, w, _ = flows.shape
    warped_events = torch.zeros(b, n, d + 1, 5, dtype=events.dtype, device=events.device)

    def out_of_bounds(x, y):
        return x < 0 or x >= w - 1 or y < 0 or y >= h - 1

    def trilinear_interpolation(x, y, z, flows):
        # determine voxel indices
        x0, y0, z0 = floor(x), floor(y), floor(z)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # compute weights
        dx, dy, dz = x - x0, y - y0, z - z0
        wx0, wy0, wz0 = 1 - dx, 1 - dy, 1 - dz
        wx1, wy1, wz1 = dx, dy, dz

        # compute flow
        # make sure indices are within bounds because we warp to edge of z
        flow = 0
        if 0 <= x0 < w and 0 <= y0 < h and 0 <= z0 < d:
            flow += flows[z0, y0, x0] * wx0 * wy0 * wz0
        if 0 <= x1 < w and 0 <= y0 < h and 0 <= z0 < d:
            flow += flows[z0, y0, x1] * wx1 * wy0 * wz0
        if 0 <= x0 < w and 0 <= y1 < h and 0 <= z0 < d:
            flow += flows[z0, y1, x0] * wx0 * wy1 * wz0
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= z0 < d:
            flow += flows[z0, y1, x1] * wx1 * wy1 * wz0
        if 0 <= x0 < w and 0 <= y0 < h and 0 <= z1 < d:
            flow += flows[z1, y0, x0] * wx0 * wy0 * wz1
        if 0 <= x1 < w and 0 <= y0 < h and 0 <= z1 < d:
            flow += flows[z1, y0, x1] * wx1 * wy0 * wz1
        if 0 <= x0 < w and 0 <= y1 < h and 0 <= z1 < d:
            flow += flows[z1, y1, x0] * wx0 * wy1 * wz1
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= z1 < d:
            flow += flows[z1, y1, x1] * wx1 * wy1 * wz1

        return flow

    def bilinear_interpolation(x, y, zi, flows):
        # determine voxel indices
        x0, y0 = floor(x), floor(y)
        x1, y1 = x0 + 1, y0 + 1

        # get corner flows
        f00 = flows[zi, y0, x0]
        f01 = flows[zi, y0, x1]
        f10 = flows[zi, y1, x0]
        f11 = flows[zi, y1, x1]

        # compute weights
        w00 = (y1 - y) * (x1 - x)
        w01 = (y1 - y) * (x - x0)
        w10 = (y - y0) * (x1 - x)
        w11 = (y - y0) * (x - x0)

        # compute flow
        flow = f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11

        return flow

    for bi in range(b):
        for ni in range(n):
            x, y, z, zi, val = events[bi, ni]
            z_orig = z.clone()
            is_out_of_bounds = out_of_bounds(x, y)
            if is_out_of_bounds:
                continue

            # warp forward: increasing z values
            # start with next integer z value
            for z_next in range(int(zi) + 1, d + 1):
                dz = z_next - z

                # interpolation to get flow at (x, y)
                if mode == "bilinear":
                    u, v = bilinear_interpolation(x, y, z_next - 1, flows[bi])
                else:
                    u, v = trilinear_interpolation(x, y, z, flows[bi])

                # update (x, y, z) using flow
                # scale flow by dz to account for non-integer z values
                x = x + u * dz
                y = y + v * dz
                z = z + dz

                # save warped event
                warped_events[bi, ni, z_next] = torch.stack([x, y, z, z_orig, val])

                # check if out of bounds
                if out_of_bounds(x, y):
                    is_out_of_bounds = True
                    break

            # only do if not yet out of bounds
            if not is_out_of_bounds:
                # reload original coordinates
                x, y, z, zi, val = events[bi, ni]

                # warp backward: decreasing z values
                for z_next in range(int(zi), -1, -1):
                    dz = z - z_next

                    # bilinear interpolation to get flow at (x, y)
                    if mode == "bilinear":
                        u, v = bilinear_interpolation(x, y, z_next, flows[bi])
                    else:
                        u, v = trilinear_interpolation(x, y, z, flows[bi])

                    # update (x, y, z) using flow
                    # scale flow by dz to account for non-integer z values
                    x = x - u * dz
                    y = y - v * dz
                    z = z - dz

                    # save warped event
                    warped_events[bi, ni, z_next] = torch.stack([x, y, z, z_orig, val])

                    # check if out of bounds
                    if out_of_bounds(x, y):
                        is_out_of_bounds = True
                        break

            # set all values to zero if out of bounds at some point
            if is_out_of_bounds:
                warped_events[bi, ni, :, -1] = 0

    return warped_events


def trilinear_splat_torch(events, grid_resolution):
    """
    Trilinearly splats events into a grid.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, z_orig, val).
        grid_resolution (tuple): The resolution of the output grid (d, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, d, h, w) with the splatted values.
    """
    b, n, _ = events.shape
    d, h, w = grid_resolution
    output = torch.zeros(b, d, h, w, dtype=events.dtype, device=events.device)

    for bi in range(b):
        for ni in range(n):
            x, y, z, val = events[bi, ni]

            # determine voxel indices
            x0, y0, z0 = floor(x), floor(y), floor(z)
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

            # compute weights
            dx, dy, dz = x - x0, y - y0, z - z0
            wx0, wy0, wz0 = 1 - dx, 1 - dy, 1 - dz
            wx1, wy1, wz1 = dx, dy, dz

            # make sure indices are within bounds
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= z0 < d:
                output[bi, z0, y0, x0] += val * wx0 * wy0 * wz0
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z0 < d:
                output[bi, z0, y0, x1] += val * wx1 * wy0 * wz0
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[bi, z0, y1, x0] += val * wx0 * wy1 * wz0
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[bi, z0, y1, x1] += val * wx1 * wy1 * wz0
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[bi, z1, y0, x0] += val * wx0 * wy0 * wz1
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[bi, z1, y0, x1] += val * wx1 * wy0 * wz1
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[bi, z1, y1, x0] += val * wx0 * wy1 * wz1
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[bi, z1, y1, x1] += val * wx1 * wy1 * wz1

    return output


"""
NOTE:
- Mode 'bilinear' doesn't have gradients for (1.1, 1.1, 0.1), 'trilinear' does
- I think trilinear will give much better grad flow, many invisible spots where weights balance each other out
- On the upper border of the image = out of bounds
- If z is integer, forward warp will not put it in the current bin, only the backward will, else number of warped events > events
"""


if __name__ == "__main__":
    methods = {
        "torch": [iterative_3d_warp_torch, trilinear_splat_torch],
        "cuda": [iterative_3d_warp_cuda, trilinear_splat_cuda],
    }
    grads, losses = [], []
    seed = torch.randint(0, 1000, (1,)).item()
    print(f"Seed: {seed}")
    for name, functions in methods.items():
        torch.manual_seed(seed)
        # n = 1
        n = 100
        # b, d, h, w = 1, 4, 6, 6
        b, d, h, w = 1, 3, 5, 5
        # events = torch.tensor([[[1.0, 1.0, 3.0, 2, 1.0]]], device="cuda")  # (b, n, 5): x, y, z, zi, val
        events = torch.rand((b, n, 5), device="cuda") * torch.tensor([w - 1, h - 1, d, d - 1, 1.0], device="cuda")
        events[..., 3] = events[..., 2].floor()
        # flows = torch.zeros((b, d, h, w, 2), device="cuda")  # (b, d, h, w, 2): u, v flow from z to z+1
        # flows[0, 1, 1, 1, 0] = 1
        # flows[0, 2, 1, 2, 0] = 1
        # flows[0, 3, 1, 3, 0] = 1
        flow_mag = torch.tensor([0.5, 0.0], device="cuda")
        flows = torch.rand((b, 1, h, w, 2), device="cuda").repeat(1, d, 1, 1, 1) * flow_mag
        flows.requires_grad = True
        visualize_tensor(flows[..., 0].detach(), title=f"x flow field {name}", folder="figures/test_warp_events")

        warp_fn, splat_fn = functions

        warped_events = warp_fn(events, flows)  # no mode trilinear yet for cuda
        print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
        print(f"Warped events ({name}) with shape {tuple(warped_events.shape)}:\n{warped_events}\n")

        warped_events = warped_events[..., [0, 1, 2, 4]].view(b, -1, 4)  # remove z_orig
        splatted = splat_fn(warped_events, (d + 1, h, w))
        visualize_tensor(splatted.detach(), title=f"splatted image {name}", folder="figures/test_warp_events")

        loss = splatted.diff(dim=1).abs()
        visualize_tensor(loss.detach(), title=f"loss image {name}", folder="figures/test_warp_events")

        loss_val = loss.sum()
        print(f"Loss val ({name}): {loss_val.item()}")
        loss_val.backward()
        losses.append(loss_val.clone())

        visualize_tensor(flows.grad[..., 0], title=f"grad x flow field {name}", folder="figures/test_warp_events")
        visualize_tensor(flows.grad[..., 1], title=f"grad y flow field {name}", folder="figures/test_warp_events")
        grads.append(flows.grad.clone())

    print(f"Losses all equal: {torch.allclose(*losses)}, largest diff: {torch.max(torch.abs(losses[0] - losses[1]))}")
    print(f"Grads all equal: {torch.allclose(*grads)}, largest diff: {torch.max(torch.abs(grads[0] - grads[1]))}")
