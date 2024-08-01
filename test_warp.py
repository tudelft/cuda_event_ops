from math import ceil, floor

import torch

from iterative_3d_warp import iterative_3d_warp  # import after torch


def iterative_3d_warp_torch(events, flows):
    """
    Iteratively warps events in 3D using flow fields and bilinear interpolation.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 4), where each event has (x, y, z, value).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).
    
    Returns:
        torch.Tensor: A tensor of shape (b, n, d + 1, 5), where each event has (x, y, z, z_orig, value).
    """
    b, n, _ = events.shape
    _, d, h, w, _ = flows.shape
    warped_events = torch.zeros(b, n, d + 1, 5, dtype=events.dtype, device=events.device)

    def out_of_bounds(x, y):
        return x < 0 or x >= w - 1 or y < 0 or y >= h - 1
    
    def bilinear_interpolation(x, y, z0, flows):
        x0, y0 = floor(x), floor(y)
        x1, y1 = x0 + 1, y0 + 1
        u00, v00 = flows[z0, y0, x0]
        u01, v01 = flows[z0, y0, x1]
        u10, v10 = flows[z0, y1, x0]
        u11, v11 = flows[z0, y1, x1]
        w00 = (x1 - x) * (y1 - y)
        w01 = (x - x0) * (y1 - y)
        w10 = (x1 - x) * (y - y0)
        w11 = (x - x0) * (y - y0)
        u = u00 * w00 + u01 * w01 + u10 * w10 + u11 * w11
        v = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
        return u, v
    
    for batch_idx in range(b):
        for event_idx in range(n):
            x, y, z, value = events[batch_idx, event_idx].clone()
            original_z = z.clone()
            is_out_of_bounds = out_of_bounds(x, y)
            if is_out_of_bounds:
                continue

            # warp forward: increasing z values
            # start with next integer z value
            for z1 in torch.arange(ceil(z), d + 1, dtype=torch.long, device=events.device):
                z0 = floor(z)
                dz = z1 - z

                # bilinear interpolation to get flow at (x, y)
                u, v = bilinear_interpolation(x, y, z0, flows[batch_idx])

                # update (x, y, z) using flow
                # scale flow by dz to account for non-integer z values
                # TODO: needs to be out of place because view created somewhere?
                x = x + u * dz
                y = y + v * dz
                z = z1

                # save warped event
                warped_events[batch_idx, event_idx, z1] = torch.stack([x, y, z, original_z, value])
            
                # check if out of bounds
                if out_of_bounds(x, y):
                    is_out_of_bounds = True
                    break
                
            # only do if not out of bounds
            if not is_out_of_bounds:
                # reload original coordinates
                x, y, z, _ = events[batch_idx, event_idx].clone()

                # warp backward: decreasing z values
                # start with previous integer z value
                for z1 in torch.arange(floor(z), -1, -1, dtype=torch.long, device=events.device):
                    z0 = ceil(z)
                    dz = z - z1

                    # bilinear interpolation to get flow at (x, y)
                    u, v = bilinear_interpolation(x, y, z0, flows[batch_idx])

                    # update (x, y, z) using flow
                    # scale flow by dz to account for non-integer z values
                    x = x - u * dz
                    y = y - v * dz
                    z = z1

                    # save warped event
                    warped_events[batch_idx, event_idx, z1] = torch.stack([x, y, z, original_z, value])
            
                    # check if out of bounds
                    if out_of_bounds(x, y):
                        is_out_of_bounds = True
                        break
                    
            # set all values to zero if out of bounds at some point
            if is_out_of_bounds:
                warped_events[batch_idx, event_idx, :, -1] = 0
        
    return warped_events


if __name__ == "__main__":
    # test 1: deterministic xyz and flow
    flow_mag = 0.5
    b, d, h, w = 1, 3, 5, 5
    events = torch.tensor([[[1, 1, 1.5, 0.9]]], device="cuda")  # (b, n, 4): x, y, z, val
    flows = torch.ones(b, d, h, w, 2, device="cuda") * flow_mag  # (b, d, h, w, 2): u, v flow from z to z+1

    warped_events_cuda = iterative_3d_warp(events, flows)  # (b, n, d + 1, 5): x, y, z, z_orig, val
    warped_events_torch = iterative_3d_warp_torch(events, flows)
    print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
    print(f"Flow with shape {tuple(flows.shape)} and magnitude {flow_mag}\n")
    print(f"Warped events (cuda) with shape {tuple(warped_events_cuda.shape)}:\n{warped_events_cuda}\n")
    print(f"Warped events (torch) with shape {tuple(warped_events_torch.shape)}:\n{warped_events_torch}\n")
    assert torch.allclose(warped_events_cuda, warped_events_torch)

    # test 2: random float xyz and flow
    n = 1
    b, d, h, w = 2, 3, 10, 10
    events = torch.rand((b, n, 4), device="cuda") * torch.tensor([w - 1, h - 1, d - 1, 1.0], device="cuda")
    flows = torch.rand((b, d, h, w, 2), device="cuda")

    warped_events_cuda = iterative_3d_warp(events, flows)
    warped_events_torch = iterative_3d_warp_torch(events, flows)
    print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
    print(f"Flow with shape {tuple(flows.shape)}\n")
    print(f"Warped events (cuda) with shape {tuple(warped_events_cuda.shape)}:\n{warped_events_cuda}\n")
    print(f"Warped events (torch) with shape {tuple(warped_events_torch.shape)}:\n{warped_events_torch}\n")
    assert torch.allclose(warped_events_cuda, warped_events_torch)
