from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def visualize_tensor(tensor, title="", figsize=(6, 2), folder="assets"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    b, h, w = tensor.shape
    tensor = tensor.view(b, h, w)

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, (b, 1), axes_pad=0.1)
    for i in range(b):
        grid[i].imshow(tensor[i].cpu().numpy())
        grid[i].axis("off")
    
    # add pixel values as text
    for i in range(b):
        for j in range(h):
            for k in range(w):
                grid[i].text(k, j, f"{tensor[i, j, k]:.3f}", color="red", fontsize=6, ha="center", va="center")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(folder / f"{title.replace(' ', '_')}.png", dpi=300)


def single_2d_warp(point, flow_field):

    def bilinear_interpolation(x, y, flow_field):
        x0, y0 = floor(x), floor(y)
        x1, y1 = x0 + 1, y0 + 1

        f00 = flow_field[y0, x0]
        f01 = flow_field[y0, x1]
        f10 = flow_field[y1, x0]
        f11 = flow_field[y1, x1]

        w00 = (y1 - y) * (x1 - x)
        w01 = (y1 - y) * (x - x0)
        w10 = (y - y0) * (x1 - x)
        w11 = (y - y0) * (x - x0)

        flow = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        return flow
    
    x, y = point
    flow = bilinear_interpolation(x, y, flow_field)
    warped_point = point + flow

    return warped_point


class Single2dWarpCustom(torch.autograd.Function):

    @staticmethod
    def forward(ctx, point, flow_field):
        ctx.save_for_backward(point, flow_field)
        return single_2d_warp(point, flow_field)

    @staticmethod
    def backward(ctx, grad_output):
        point, flow_field = ctx.saved_tensors

        x, y = point
        x0, y0 = floor(x), floor(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0

        # gradients wrt point
        f00 = flow_field[y0, x0]
        f01 = flow_field[y0, x1]
        f10 = flow_field[y1, x0]
        f11 = flow_field[y1, x1]

        dflow_dx = (f01 - f00) * (1 - dy) + (f11 - f10) * dy
        dflow_dy = (f10 - f00) * (1 - dx) + (f11 - f01) * dx
        
        dflow_dpoint = torch.stack([dflow_dx, dflow_dy], dim=-1)
        grad_point = grad_output.view(-1, 1) * (torch.eye(2, device=grad_output.device) + dflow_dpoint)

        # gradients wrt flow field
        grad_flow_field = torch.zeros_like(flow_field)
        
        # here we distribute the gradient from the flow field to the corresponding weights
        grad_flow_field[y0, x0] += grad_output * (1 - dx) * (1 - dy)
        grad_flow_field[y0, x1] += grad_output * dx * (1 - dy)
        grad_flow_field[y1, x0] += grad_output * (1 - dx) * dy
        grad_flow_field[y1, x1] += grad_output * dx * dy

        return grad_point, grad_flow_field


def single_2d_warp_custom(point, flow_field):
    return Single2dWarpCustom.apply(point, flow_field)


def iterative_2d_warp(points, flow_field, warps, custom=False):
    b, n, _ = points.shape
    warped_points = []

    for bi in range(b):
        for ni in range(n):
            point = points[bi, ni]
            for _ in range(warps):
                if not custom:
                    warped_point = single_2d_warp(point, flow_field[bi])
                else:
                    warped_point = single_2d_warp_custom(point, flow_field[bi])
                warped_points.append(warped_point)
                point = warped_point
    
    warped_points = torch.stack(warped_points).view(b, n * warps, 2)

    return warped_points


def bilinear_splat(points, grid_resolution):
    b, n, _ = points.shape
    h, w = grid_resolution
    output = torch.zeros(b, h, w, dtype=points.dtype, device=points.device)
    val = 1

    for bi in range(b):
        for ni in range(n):
            x, y = points[bi, ni]

            x0, y0 = floor(x), floor(y)
            x1, y1 = x0 + 1, y0 + 1

            xd, yd = x - x0, y - y0
            wx0, wy0 = 1 - xd, 1 - yd
            wx1, wy1 = xd, yd

            if 0 <= x0 < w and 0 <= y0 < h:
                output[bi, y0, x0] += val * wx0 * wy0
            if 0 <= x0 < w and 0 <= y1 < h:
                output[bi, y1, x0] += val * wx0 * wy1
            if 0 <= x1 < w and 0 <= y0 < h:
                output[bi, y0, x1] += val * wx1 * wy0
            if 0 <= x1 < w and 0 <= y1 < h:
                output[bi, y1, x1] += val * wx1 * wy1

    return output


if __name__ == "__main__":
    n = 1
    b, h, w = 1, 5, 5
    warps = 3
    points = torch.tensor([[[1.0, 1.0]]], device="cuda")
    flow_field = torch.zeros((b, h, w, 2), device="cuda")
    flow_field[0, 1, 1, 0] = 1
    flow_field[0, 1, 2, 0] = 1
    flow_field[0, 1, 3, 0] = 1
    flow_field_custom = flow_field.clone()
    flow_field.requires_grad = True
    flow_field_custom.requires_grad = True
    visualize_tensor(flow_field[..., 0].detach(), title="x flow field")

    warped_points = iterative_2d_warp(points, flow_field, warps)
    warped_points_custom = iterative_2d_warp(points, flow_field_custom, warps, custom=True)
    print(f"Original points with shape {tuple(points.shape)}:\n{points}\n")
    print(f"Warped points with shape {tuple(warped_points.shape)}:\n{warped_points}\n")
    print(f"Warped points (custom) with shape {tuple(warped_points_custom.shape)}:\n{warped_points_custom}\n")

    splatted = bilinear_splat(warped_points, (h, w))
    splatted_custom = bilinear_splat(warped_points_custom, (h, w))
    visualize_tensor(splatted.detach(), title="splatted image")
    visualize_tensor(splatted_custom.detach(), title="custom splatted image")

    loss = splatted.diff(dim=1).abs()
    loss_custom = splatted_custom.diff(dim=1).abs()
    visualize_tensor(loss.detach(), title="loss image")
    visualize_tensor(loss_custom.detach(), title="loss image")

    loss_val = loss.sum()
    loss_val_custom = loss_custom.sum()
    print(f"Loss val: {loss_val.item()}\n")
    print(f"Loss val (custom): {loss_val_custom.item()}\n")
    flow_field.grad = None
    flow_field_custom.grad = None
    loss_val.backward()
    loss_val_custom.backward()

    visualize_tensor(flow_field.grad[..., 0], title="grad x flow field")
    visualize_tensor(flow_field.grad[..., 1], title="grad y flow field")
    visualize_tensor(flow_field_custom.grad[..., 0], title="custom grad x flow field")
    visualize_tensor(flow_field_custom.grad[..., 1], title="custom grad y flow field")
