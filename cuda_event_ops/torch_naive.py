from math import floor

import torch


def iterative_3d_warp(events, flows):
    """
    Iteratively warps events in 3D using bilinearly-sampled flows.

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

    # go through all events
    for bi in range(b):
        for ni in range(n):
            # load event
            x, y, z, zi, val = events[bi, ni]
            z_orig = z.clone()

            # skip if value is zero
            if val == 0:
                continue

            # skip if already out of bounds
            is_out_of_bounds = out_of_bounds(x, y)
            if is_out_of_bounds:
                continue

            # warp forward: increasing z values
            # start with next integer z value
            for z_next in range(int(zi) + 1, d + 1):
                dz = z_next - z

                # interpolation to get flow at (x, y)
                u, v = bilinear_interpolation(x, y, z_next - 1, flows[bi])

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
                    u, v = bilinear_interpolation(x, y, z_next, flows[bi])

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


def trilinear_splat(events, grid_resolution):
    """
    Trilinearly splats events into a 3D grid.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, z_orig, val).
        grid_resolution (tuple): The resolution of the output grid (d, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, d, h, w) with the splatted values.
    """
    b, n, _ = events.shape
    d, h, w = grid_resolution
    splatted = torch.zeros(b, d, h, w, dtype=events.dtype, device=events.device)

    # go through all events
    for bi in range(b):
        for ni in range(n):
            # load event
            x, y, z, val = events[bi, ni]

            # skip if value is zero
            if val == 0:
                continue

            # determine voxel indices
            x0, y0, z0 = floor(x), floor(y), floor(z)
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

            # compute weights
            dx, dy, dz = x - x0, y - y0, z - z0
            wx0, wy0, wz0 = 1 - dx, 1 - dy, 1 - dz
            wx1, wy1, wz1 = dx, dy, dz

            # compute conditions
            x0_in = 0 <= x0 < w
            x1_in = 0 <= x1 < w
            y0_in = 0 <= y0 < h
            y1_in = 0 <= y1 < h
            z0_in = 0 <= z0 < d
            z1_in = 0 <= z1 < d

            # add value if corner within bounds
            if x0_in and y0_in and z0_in:
                splatted[bi, z0, y0, x0] += val * wx0 * wy0 * wz0
            if x1_in and y0_in and z0_in:
                splatted[bi, z0, y0, x1] += val * wx1 * wy0 * wz0
            if x0_in and y1_in and z0_in:
                splatted[bi, z0, y1, x0] += val * wx0 * wy1 * wz0
            if x1_in and y1_in and z0_in:
                splatted[bi, z0, y1, x1] += val * wx1 * wy1 * wz0
            if x0_in and y0_in and z1_in:
                splatted[bi, z1, y0, x0] += val * wx0 * wy0 * wz1
            if x1_in and y0_in and z1_in:
                splatted[bi, z1, y0, x1] += val * wx1 * wy0 * wz1
            if x0_in and y1_in and z1_in:
                splatted[bi, z1, y1, x0] += val * wx0 * wy1 * wz1
            if x1_in and y1_in and z1_in:
                splatted[bi, z1, y1, x1] += val * wx1 * wy1 * wz1

    return splatted
