import torch

from cuda_3d_ops import trilinear_splat_cuda
from test_warp_events import trilinear_splat_torch


if __name__ == "__main__":
    b, d, h, w = 1, 3, 5, 5

    methods = {"torch": trilinear_splat_torch, "cuda": trilinear_splat_cuda}
    for name, fn in methods.items():
        warped_events = torch.tensor([[
            [1, 1, 0, 1],
            [2, 1, 1, 1],
            [3, 1, 2, 1],
            [4, 1, 3, 1],
            ]],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        splatted = fn(warped_events, (d + 1, h, w))
        splatted.register_hook(lambda grad: print(f"Gradient splatted ({name}): {grad}\n"))
        loss = splatted.diff(dim=1).abs().sum()
        loss.backward()

        print(f"Loss ({name}): {loss.item()}\n")
        print(f"Warped events ({name}):\n{warped_events}\n")
        # print(f"Splatted events ({name}):\n{splatted}\n")
        print(f"Gradient ({name}):\n{warped_events.grad}\n")
