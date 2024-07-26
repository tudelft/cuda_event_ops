import torch

from cuda_3d_ops import iterative_3d_warp  # import after torch


# test 1: deterministic xyz and flow
flow_mag = 0.5
d, h, w = 3, 5, 5
events = torch.tensor([[1, 1, 1.5, 0.9]], device="cuda")  # x, y, z, val
flows = torch.ones(d, h, w, 2, device="cuda") * flow_mag  # u, v flow from z to z+1
xy_bounds = (0, w, 0, h)

warped_events = iterative_3d_warp(events, flows, *xy_bounds)
print(f"Original events:\n{events}\n")
print(f"Flow: {flow_mag}\n")
print(f"Warped events:\n{warped_events}\n")

# test 2: random float xyz and flow
n = 1
d, h, w = 3, 10, 10
events = torch.rand((n, 4), device="cuda") * torch.tensor([w, h, d, 1.0], device="cuda")  # x, y, z, val
flows = torch.rand((d, h, w, 2), device="cuda")  # u, v flow from z to z+1
xy_bounds = (0, w, 0, h)

warped_events = iterative_3d_warp(events, flows, *xy_bounds)
print(f"Original events:\n{events}\n")
print(f"Warped events:\n{warped_events}")
