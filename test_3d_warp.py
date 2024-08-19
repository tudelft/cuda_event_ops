from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def visualize_tensor(tensor, title="", figsize=(6, 2), folder="figures/test_3d_warp"):
    """
    Visualizes a tensor with shape (b, d, h, w) as a grid of images.

    Args:
        tensor (torch.Tensor): A tensor with shape (b, d, h, w).
        title (str): The title of the plot.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    b, d, h, w = tensor.shape
    tensor = tensor.view(b * d, h, w)

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, (b, d), axes_pad=0.1)
    for i in range(b * d):
        grid[i].imshow(tensor[i].cpu().numpy())
        grid[i].axis("off")
    
    # add pixel values as text
    for i in range(b * d):
        for j in range(h):
            for k in range(w):
                grid[i].text(k, j, f"{tensor[i, j, k]:.3f}", color="red", fontsize=6, ha="center", va="center")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(folder / f"{title.replace(' ', '_')}.png", dpi=300)


def single_3d_warp(point, flow_field):

    def trilinear_interpolation(x, y, z, flow_field):
        # determine voxel indices
        x0, y0, z0 = floor(x), floor(y), floor(z)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # get corner flows
        f000 = flow_field[z0, y0, x0]
        f001 = flow_field[z0, y0, x1]
        f010 = flow_field[z0, y1, x0]
        f011 = flow_field[z0, y1, x1]

        f100 = flow_field[z1, y0, x0]
        f101 = flow_field[z1, y0, x1]
        f110 = flow_field[z1, y1, x0]
        f111 = flow_field[z1, y1, x1]

        # compute weights
        w000 = (z1 - z) * (y1 - y) * (x1 - x)
        w001 = (z1 - z) * (y1 - y) * (x - x0)
        w010 = (z1 - z) * (y - y0) * (x1 - x)
        w011 = (z1 - z) * (y - y0) * (x - x0)

        w100 = (z - z0) * (y1 - y) * (x1 - x)
        w101 = (z - z0) * (y1 - y) * (x - x0)
        w110 = (z - z0) * (y - y0) * (x1 - x)
        w111 = (z - z0) * (y - y0) * (x - x0)

        # compute flow
        flow = (
            w000 * f000 +
            w001 * f001 +
            w010 * f010 +
            w011 * f011 +
            w100 * f100 +
            w101 * f101 +
            w110 * f110 +
            w111 * f111
        )
        return flow
    
    x, y, z = point
    flow = trilinear_interpolation(x, y, z, flow_field)
    warped_point = point + flow

    return warped_point


class Single3dWarpCustom(torch.autograd.Function):

    @staticmethod
    def forward(ctx, point, flow_field):
        ctx.save_for_backward(point, flow_field)
        return single_3d_warp(point, flow_field)

    @staticmethod
    def backward(ctx, grad_output):
        point, flow_field = ctx.saved_tensors

        x, y, z = point
        x0, y0, z0 = floor(x), floor(y), floor(z)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        dx, dy, dz = x - x0, y - y0, z - z0

        # gradients wrt point
        f000 = flow_field[z0, y0, x0]
        f001 = flow_field[z0, y0, x1]
        f010 = flow_field[z0, y1, x0]
        f011 = flow_field[z0, y1, x1]

        f100 = flow_field[z1, y0, x0]
        f101 = flow_field[z1, y0, x1]
        f110 = flow_field[z1, y1, x0]
        f111 = flow_field[z1, y1, x1]

        dflow_dx = (
            (f001 - f000) * (1 - dy) * (1 - dz) +
            (f011 - f010) * dy * (1 - dz) +
            (f101 - f100) * (1 - dy) * dz +
            (f111 - f110) * dy * dz
        )

        dflow_dy = (
            (f010 - f000) * (1 - dx) * (1 - dz) +
            (f011 - f001) * dx * (1 - dz) +
            (f110 - f100) * (1 - dx) * dz +
            (f111 - f101) * dx * dz
        )
        
        dflow_dz = (
            (f100 - f000) * (1 - dx) * (1 - dy) +
            (f101 - f001) * dx * (1 - dy) +
            (f110 - f010) * (1 - dx) * dy +
            (f111 - f011) * dx * dy
        )
        
        dflow_dpoint = torch.stack([dflow_dx, dflow_dy, dflow_dz], dim=-1)
        dwarped_point_dpoint = torch.eye(3, device=grad_output.device) + dflow_dpoint
        grad_point = grad_output.view(-1, 1) * dwarped_point_dpoint
        grad_point = grad_point.sum(0)  # sum over rows to get aggregate output grad wrt single input dim

        # gradients wrt flow field
        grad_flow_field = torch.zeros_like(flow_field)
        
        # here we distribute the gradient from the flow field to the corresponding weights
        grad_flow_field[z0, y0, x0] += grad_output * (1 - dx) * (1 - dy) * (1 - dz)
        grad_flow_field[z0, y0, x1] += grad_output * dx * (1 - dy) * (1 - dz)
        grad_flow_field[z0, y1, x0] += grad_output * (1 - dx) * dy * (1 - dz)
        grad_flow_field[z0, y1, x1] += grad_output * dx * dy * (1 - dz)

        grad_flow_field[z1, y0, x0] += grad_output * (1 - dx) * (1 - dy) * dz
        grad_flow_field[z1, y0, x1] += grad_output * dx * (1 - dy) * dz
        grad_flow_field[z1, y1, x0] += grad_output * (1 - dx) * dy * dz
        grad_flow_field[z1, y1, x1] += grad_output * dx * dy * dz

        return grad_point, grad_flow_field


def single_3d_warp_custom(point, flow_field):
    return Single3dWarpCustom.apply(point, flow_field)


def iterative_3d_warp(points, flow_field, warps, custom=False):
    b, n, _ = points.shape
    warped_points = []

    for bi in range(b):
        for ni in range(n):
            point = points[bi, ni]
            for _ in range(warps):
                if not custom:
                    warped_point = single_3d_warp(point, flow_field[bi])
                else:
                    warped_point = single_3d_warp_custom(point, flow_field[bi])
                warped_points.append(warped_point)
                point = warped_point
    
    warped_points = torch.stack(warped_points).view(b, n * warps, 3)

    return warped_points


class Iterative3dWarpCustom(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_field, warps):
        warped_points = iterative_3d_warp(points, flow_field, warps, custom=True)
        ctx.save_for_backward(warped_points, points, flow_field)
        ctx.warps = warps
        return warped_points

    @staticmethod
    def backward(ctx, grad_output):
        warped_points, points, flow_field = ctx.saved_tensors
        warps = ctx.warps
        b, n, _ = points.shape
        grad_flow_field = torch.zeros_like(flow_field)

        for bi in range(b):
            for ni in range(n):
                grad_warped_point, grad_point = 0, 0
                for wi in reversed(range(warps)):
                    grad_warped_point += grad_output[bi, ni * warps + wi]

                    # get point and corners before warp
                    x, y, z = warped_points[bi, ni * warps + wi - 1] if wi > 0 else points[bi, ni]
                    x0, y0, z0 = floor(x), floor(y), floor(z)
                    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                    dx, dy, dz = x - x0, y - y0, z - z0

                    # gradients wrt flow field
                    # here we distribute the gradient from the flow field to the corresponding weights
                    grad_flow_field[bi, z0, y0, x0] += grad_warped_point * (1 - dx) * (1 - dy) * (1 - dz)
                    grad_flow_field[bi, z0, y0, x1] += grad_warped_point * dx * (1 - dy) * (1 - dz)
                    grad_flow_field[bi, z0, y1, x0] += grad_warped_point * (1 - dx) * dy * (1 - dz)
                    grad_flow_field[bi, z0, y1, x1] += grad_warped_point * dx * dy * (1 - dz)

                    grad_flow_field[bi, z1, y0, x0] += grad_warped_point * (1 - dx) * (1 - dy) * dz
                    grad_flow_field[bi, z1, y0, x1] += grad_warped_point * dx * (1 - dy) * dz
                    grad_flow_field[bi, z1, y1, x0] += grad_warped_point * (1 - dx) * dy * dz
                    grad_flow_field[bi, z1, y1, x1] += grad_warped_point * dx * dy * dz

                    # gradients wrt point
                    f000 = flow_field[bi, z0, y0, x0]
                    f001 = flow_field[bi, z0, y0, x1]
                    f010 = flow_field[bi, z0, y1, x0]
                    f011 = flow_field[bi, z0, y1, x1]

                    f100 = flow_field[bi, z1, y0, x0]
                    f101 = flow_field[bi, z1, y0, x1]
                    f110 = flow_field[bi, z1, y1, x0]
                    f111 = flow_field[bi, z1, y1, x1]

                    dflow_dx = (
                        (f001 - f000) * (1 - dy) * (1 - dz) +
                        (f011 - f010) * dy * (1 - dz) +
                        (f101 - f100) * (1 - dy) * dz +
                        (f111 - f110) * dy * dz
                    )

                    dflow_dy = (
                        (f010 - f000) * (1 - dx) * (1 - dz) +
                        (f011 - f001) * dx * (1 - dz) +
                        (f110 - f100) * (1 - dx) * dz +
                        (f111 - f101) * dx * dz
                    )
                    
                    dflow_dz = (
                        (f100 - f000) * (1 - dx) * (1 - dy) +
                        (f101 - f001) * dx * (1 - dy) +
                        (f110 - f010) * (1 - dx) * dy +
                        (f111 - f011) * dx * dy
                    )
                    
                    dflow_dpoint = torch.stack([dflow_dx, dflow_dy, dflow_dz], dim=-1)
                    dwarped_point_dpoint = torch.eye(3, device=grad_output.device) + dflow_dpoint
                    grad_point = (grad_warped_point.view(-1, 1) * dwarped_point_dpoint).sum(0)
                    grad_warped_point = grad_point.clone()

        return None, grad_flow_field, None


def iterative_3d_warp_custom(points, flow_field, warps):
    return Iterative3dWarpCustom.apply(points, flow_field, warps)


def trilinear_splat(points, grid_resolution):
    b, n, _ = points.shape
    d, h, w = grid_resolution
    output = torch.zeros(b, d, h, w, dtype=points.dtype, device=points.device)
    val = 1

    for bi in range(b):
        for ni in range(n):
            x, y, z = points[bi, ni]

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
From here to warping events:
- Z is actually time, flows should be scaled by dt to new time
- Bilinear interpolation for flow field based on floor of time
- Allow value
- Check for out of bounds
- Warp forward and backward
"""


if __name__ == "__main__":
    n = 1
    b, d, h, w = 1, 3, 5, 5
    warps = 3
    points = torch.tensor([[[1.1, 1.1, 0.0]]], device="cuda")
    flow_field = torch.zeros((b, d + 1, h, w, 3), device="cuda")
    flow_field[0, 0, 1, 1, 0] = 1
    flow_field[0, 1, 1, 2, 0] = 1
    flow_field[0, 2, 1, 3, 0] = 1
    flow_field[..., -1] = 1
    flow_field_custom = flow_field.clone()
    flow_field.requires_grad = True
    flow_field_custom.requires_grad = True
    visualize_tensor(flow_field[..., 0].detach(), title="x flow field")

    warped_points = iterative_3d_warp(points, flow_field, warps)
    # warped_points = iterative_3d_warp(points, flow_field, warps, custom=True)
    warped_points_custom = iterative_3d_warp_custom(points, flow_field_custom, warps)
    print(f"Original points with shape {tuple(points.shape)}:\n{points}\n")
    print(f"Warped points with shape {tuple(warped_points.shape)}:\n{warped_points}\n")
    print(f"Warped points (custom) with shape {tuple(warped_points_custom.shape)}:\n{warped_points_custom}\n")

    splatted = trilinear_splat(warped_points, (d + 1, h, w))
    splatted_custom = trilinear_splat(warped_points_custom, (d + 1, h, w))
    visualize_tensor(splatted.detach(), title="splatted image")
    visualize_tensor(splatted_custom.detach(), title="custom splatted image")

    loss = splatted.diff(dim=1).abs()
    loss_custom = splatted_custom.diff(dim=1).abs()
    visualize_tensor(loss.detach(), title="loss image")
    visualize_tensor(loss_custom.detach(), title="loss image")

    loss_val = loss.sum()
    loss_val_custom = loss_custom.sum()
    print(f"Loss val: {loss_val.item()}")
    print(f"Loss val (custom): {loss_val_custom.item()}")
    loss_val.backward()
    loss_val_custom.backward()

    visualize_tensor(flow_field.grad[..., 0], title="grad x flow field")
    visualize_tensor(flow_field.grad[..., 1], title="grad y flow field")
    visualize_tensor(flow_field_custom.grad[..., 0], title="custom grad x flow field")
    visualize_tensor(flow_field_custom.grad[..., 1], title="custom grad y flow field")

    print(f"Grads all equal: {torch.allclose(flow_field.grad, flow_field_custom.grad)}")
