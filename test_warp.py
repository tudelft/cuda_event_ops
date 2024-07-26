import torch

from cuda_3d_ops import iterative_3d_warp  # import after torch


# test 1: deterministic xyz and flow
flow_mag = 0.5
b, d, h, w = 1, 3, 5, 5
events = torch.tensor([[[1, 1, 1.5, 0.9]]], device="cuda")  # (b, n, 4): x, y, z, val
flows = torch.ones(b, d, h, w, 2, device="cuda") * flow_mag  # (b, d, h, w, 2): u, v flow from z to z+1
xy_bounds = (0, w, 0, h)

warped_events = iterative_3d_warp(events, flows, *xy_bounds)  # (b, n, d + 1, 5): x, y, z, z_orig, val
print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
print(f"Flow with shape {tuple(flows.shape)} and magnitude {flow_mag}\n")
print(f"Warped events with shape {tuple(warped_events.shape)}:\n{warped_events}\n")

# test 2: random float xyz and flow
n = 1
b, d, h, w = 2, 3, 10, 10
events = torch.rand((b, n, 4), device="cuda") * torch.tensor([w, h, d, 1.0], device="cuda")
flows = torch.rand((b, d, h, w, 2), device="cuda")
xy_bounds = (0, w, 0, h)

warped_events = iterative_3d_warp(events, flows, *xy_bounds)
print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
print(f"Flow with shape {tuple(flows.shape)}\n")
print(f"Warped events with shape {tuple(warped_events.shape)}:\n{warped_events}\n")
