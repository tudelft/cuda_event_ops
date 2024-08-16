from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def visualize_tensor(tensor, title="", figsize=(6, 2), folder="assets"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    b, w = tensor.shape
    tensor = tensor.view(b, 1, w)

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, (b, 1), axes_pad=0.1)
    for i in range(b):
        grid[i].imshow(tensor[i].cpu().numpy())
        grid[i].axis("off")
    
    # add pixel values as text
    for i in range(b):
        for j in range(1):
            for k in range(w):
                grid[i].text(k, j, f"{tensor[i, j, k]:.3f}", color="red", fontsize=6, ha="center", va="center")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(folder / f"{title.replace(' ', '_')}.png", dpi=300)


def single_1d_warp(point, flow_field):

    def linear_interpolation(x, flow_field):
        x0 = floor(x)
        x1 = x0 + 1

        f0 = flow_field[x0]
        f1 = flow_field[x1]

        w0 = (x1 - x)
        w1 = (x - x0)

        flow = w0 * f0 + w1 * f1
        return flow
    
    (x,) = point
    flow = linear_interpolation(x, flow_field)
    warped_point = point + flow

    return warped_point


class Single1dWarpCustom(torch.autograd.Function):

    @staticmethod
    def forward(ctx, point, flow_field):
        ctx.save_for_backward(point, flow_field)
        return single_1d_warp(point, flow_field)

    @staticmethod
    def backward(ctx, grad_output):
        point, flow_field = ctx.saved_tensors

        (x,) = point
        x0 = floor(x)
        x1 = x0 + 1

        # gradients wrt point
        f0 = flow_field[x0]
        f1 = flow_field[x1]

        dflow_dx = f1 - f0

        grad_x = grad_output * (1 + dflow_dx)
        grad_point = torch.stack([grad_x[0]])

        # gradients wrt flow field
        grad_flow_field = torch.zeros_like(flow_field)
        
        # here we distribute the gradient from the flow field to the corresponding weights
        grad_flow_field[x0] += grad_output * (1 - (x - x0))
        grad_flow_field[x1] += grad_output * (x - x0)

        return grad_point, grad_flow_field


def single_1d_warp_custom(point, flow_field):
    return Single1dWarpCustom.apply(point, flow_field)


def iterative_1d_warp(points, flow_field, warps, custom=False):
    b, n, _ = points.shape
    warped_points = []

    for bi in range(b):
        for ni in range(n):
            point = points[bi, ni]
            for _ in range(warps):
                if not custom:
                    warped_point = single_1d_warp(point, flow_field[bi])
                else:
                    warped_point = single_1d_warp_custom(point, flow_field[bi])
                # warped_point.register_hook(lambda grad: print(grad))
                warped_points.append(warped_point)
                point = warped_point
    
    warped_points = torch.stack(warped_points).view(b, n * warps, 1)

    return warped_points


def linear_splat(points, grid_resolution):
    b, n, _ = points.shape
    (w,) = grid_resolution
    output = torch.zeros(b, w, dtype=points.dtype, device=points.device)
    val = 1

    for bi in range(b):
        for ni in range(n):
            (x,) = points[bi, ni]

            x0 = floor(x)
            x1 = x0 + 1

            xd = x - x0
            wx0 = 1 - xd
            wx1 = xd

            if 0 <= x0 < w:
                output[bi, x0] += val * wx0
            if 0 <= x1 < w:
                output[bi, x1] += val * wx1

    return output


if __name__ == "__main__":
    n = 1
    b, w = 1, 5
    warps = 3
    points = torch.tensor([[[1.0]]], device="cuda")
    flow_field = torch.zeros((b, w, 1), device="cuda")
    flow_field[0, 1, 0] = 1
    flow_field[0, 2, 0] = 1
    flow_field[0, 3, 0] = 1
    flow_field_custom = flow_field.clone()
    flow_field.requires_grad = True
    flow_field_custom.requires_grad = True
    visualize_tensor(flow_field[..., 0].detach(), title="flow field")

    warped_points = iterative_1d_warp(points, flow_field, warps)
    warped_points_custom = iterative_1d_warp(points, flow_field_custom, warps, custom=True)
    print(f"Original points with shape {tuple(points.shape)}:\n{points}\n")
    print(f"Warped points with shape {tuple(warped_points.shape)}:\n{warped_points}\n")
    print(f"Warped points (custom) with shape {tuple(warped_points_custom.shape)}:\n{warped_points_custom}\n")

    # splatted = linear_splat(warped_points, (w,))
    # splatted_custom = linear_splat(warped_points_custom, (w,))
    # visualize_tensor(splatted.detach(), title="splatted image")
    # visualize_tensor(splatted_custom.detach(), title="custom splatted image")

    # loss = splatted.diff(dim=1).abs()
    # loss_custom = splatted_custom.diff(dim=1).abs()
    # visualize_tensor(loss.detach(), title="loss image")
    # visualize_tensor(loss_custom.detach(), title="loss image")

    # loss_val = loss.sum()
    # loss_val_custom = loss_custom.sum()
    loss_val = warped_points.sum()
    loss_val_custom = warped_points_custom.sum()
    print(f"Loss val: {loss_val.item()}\n")
    print(f"Loss val (custom): {loss_val_custom.item()}\n")
    loss_val.backward()
    loss_val_custom.backward()

    visualize_tensor(flow_field.grad[..., 0], title="grad flow field")
    visualize_tensor(flow_field_custom.grad[..., 0], title="custom grad flow field")
