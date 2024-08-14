from math import ceil, floor

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def visualize_tensor(tensor, title="", figsize=(6, 2)):
    """
    Visualizes a tensor with shape (b, d, h, w) as a grid of images.

    Args:
        tensor (torch.Tensor): A tensor with shape (b, d, h, w).
        title (str): The title of the plot.
    """
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
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)


def iterative_3d_warp_simple(events, flows, autograd=True):
    """
    Iteratively warps events in 3D using flow fields and bilinear interpolation.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 4), where each event has (x, y, z, value).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).
    
    Returns:
        torch.Tensor: A tensor of shape (b, n, 4), where each warped event has (x, y, z, value).
    """
    b, n, _ = events.shape
    _, d, _, _, _ = flows.shape
    warped_events = []

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
    
    for bi in range(b):
        for ei in range(n):
            x, y, z, val = events[bi, ei]

            # warp forward: increasing z values
            # start with next integer z value
            z_ceil = ceil(z) if z != int(z) else int(z) + 1
            for z1 in torch.arange(z_ceil, d + 1, dtype=torch.long, device=events.device):
                z0 = floor(z)
                dz = z1 - z

                # bilinear interpolation to get flow at (x, y)
                u, v = bilinear_interpolation(x, y, z0, flows[bi])

                # update (x, y, z) using flow
                x = x + u * dz
                y = y + v * dz
                z = z1

                # TODO: where is the change in u and v grad compared to warped_events grad coming from?
                if autograd:
                    u.register_hook(print_grad(u, f"u grad at ({x:.3f}, {y:.3f}, {z1:.3f})"))
                    v.register_hook(print_grad(v, f"v grad at ({x:.3f}, {y:.3f}, {z1:.3f})"))

                # save warped event
                warped_event = torch.stack([x, y, z, val])
                warped_events.append(warped_event)
    
    # reshape
    warped_events = torch.stack(warped_events).view(b, n * d, 4)
            
    return warped_events


def iterative_3d_warp_simple_grad_backward(grad_output, warped_events, events, flows):
    b, n, _ = events.shape
    _, d, _, _, _ = flows.shape

    # initialize gradients
    grad_flows = torch.zeros_like(flows)

    for bi in range(b):
        for ei in range(n):
            x_orig, y_orig, z_orig, _ = events[bi, ei]
            grad_carry = 0

            for di in reversed(range(d)):
                # get positions before and after warping
                if di > 0:
                    prev_x, prev_y, prev_z, _ = warped_events[bi, ei * d + di - 1]
                    dz = 1
                else:
                    prev_x, prev_y, prev_z = x_orig, y_orig, z_orig
                    dz = warped_events[bi, ei * d + di, 2] - z_orig
                x, y, z, _ = warped_events[bi, ei * d + di]
                
                # get bilinear interpolation weights
                x0, y0 = floor(prev_x), floor(prev_y)
                x1, y1 = x0 + 1, y0 + 1

                w00 = (x1 - prev_x) * (y1 - prev_y)
                w01 = (prev_x - x0) * (y1 - prev_y)
                w10 = (x1 - prev_x) * (prev_y - y0)
                w11 = (prev_x - x0) * (prev_y - y0)

                z0 = floor(prev_z)
                # TODO: adding carry makes it correct for all flows equal
                grad_carry += grad_output[bi, ei * d + di, 0:2] * dz
                # grad_carry = grad_output[bi, ei * d + di, 0:2] * dz
                grad_flows[bi, z0, y0, x0] += grad_carry * w00
                grad_flows[bi, z0, y0, x1] += grad_carry * w01
                grad_flows[bi, z0, y1, x0] += grad_carry * w10
                grad_flows[bi, z0, y1, x1] += grad_carry * w11
                # grad_carry += -grad_output[bi, ei * d + di, 2]
                print(f"custom v grad at ({x:.3f}, {y:.3f}, {z:.3f}): {grad_carry[1]:.3f}")
                print(f"custom u grad at ({x:.3f}, {y:.3f}, {z:.3f}): {grad_carry[0]:.3f}")

    return grad_flows


class Iterative3dWarpSimple(torch.autograd.Function):
    @staticmethod
    def forward(ctx, events, flows):
        warped_events = iterative_3d_warp_simple(events, flows, autograd=False)
        ctx.save_for_backward(warped_events, events, flows)
        return warped_events
    
    @staticmethod
    def backward(ctx, grad_output):
        warped_events, events, flows = ctx.saved_tensors
        grad_flows = iterative_3d_warp_simple_grad_backward(grad_output, warped_events, events, flows)
        return None, grad_flows


def iterative_3d_warp_custom(events, flows):
    return Iterative3dWarpSimple.apply(events, flows)


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
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[batch_idx, z0, y1, x0] += value * wx0 * wy1 * wz0
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z0 < d:
                output[batch_idx, z0, y0, x1] += value * wx1 * wy0 * wz0
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z0 < d:
                output[batch_idx, z0, y1, x1] += value * wx1 * wy1 * wz0
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[batch_idx, z1, y0, x0] += value * wx0 * wy0 * wz1
            if 0 <= x0 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[batch_idx, z1, y1, x0] += value * wx0 * wy1 * wz1
            if 0 <= x1 < w and 0 <= y0 < h and 0 <= z1 < d:
                output[batch_idx, z1, y0, x1] += value * wx1 * wy0 * wz1
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= z1 < d:
                output[batch_idx, z1, y1, x1] += value * wx1 * wy1 * wz1

    return output


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def print_grad(var, name):
    def hook(grad):
        print(f"{name}: {grad:.3f}")
    return hook


def print_tensor_grad(var, name):
    def hook(grad):
        print(name)
        print(grad)
    return hook


"""
Adding gradient carry makes the gradients between manual and torch match for
the case where all flows are the same (1), but not for non-equal flows (2).

So carry needs modification in a way. When I modify carry to add -dz after,
(2) matches but (1) doesn't.

I think we need to take into account that the flow changes based on the position,
so the flow is dependent on the previous position/flow. That's why all flows equal works:
grad is zero.

Doing equal flows for all timesteps seems to suggest it's a spatial gradient.
"""

if __name__ == "__main__":
    n = 1
    b, d, h, w = 1, 3, 5, 5
    events = torch.tensor([[[1, 1, 0.0, 0.9]]], device="cuda")  # (b, n, 4): x, y, z, val
    flows = torch.zeros((b, d, h, w, 2), device="cuda")
    # flows[..., 0] = 1  # NOTE: (1)
    flows[0, 0, 1, 1, 0] = 1  # NOTE: (2)
    flows[0, 1, 1, 2, 0] = 1
    flows[0, 2, 1, 3, 0] = 1
    flows_custom = flows.clone()
    flows.requires_grad = True
    flows_custom.requires_grad = True
    visualize_tensor(flows[..., 0].detach(), title="x flow")

    warped_events = iterative_3d_warp_simple(events, flows)  # (b, n, 4): x, y, z, val
    warped_events_custom = iterative_3d_warp_custom(events, flows_custom)
    print(f"Original events with shape {tuple(events.shape)}:\n{events}\n")
    print(f"Warped events with shape {tuple(warped_events.shape)}:\n{warped_events}\n")
    print(f"Warped events (custom) with shape {tuple(warped_events_custom.shape)}:\n{warped_events_custom}\n")

    # from torchviz import make_dot
    # make_dot(warped_events, params=dict(flow=flows), show_attrs=True, show_saved=True).render("graph", format="png")

    splatted = trilinear_splat_torch(warped_events, (d + 1, h, w))
    splatted_custom = trilinear_splat_torch(warped_events_custom, (d + 1, h, w))
    visualize_tensor(splatted.detach(), title="splatted image")
    visualize_tensor(splatted_custom.detach(), title="custom splatted image")

    loss = splatted.diff(dim=1).abs()
    loss_custom = splatted_custom.diff(dim=1).abs()
    visualize_tensor(loss.detach(), title="loss image")
    visualize_tensor(loss_custom.detach(), title="custom loss image")

    warped_events.register_hook(set_grad(warped_events))
    warped_events_custom.register_hook(set_grad(warped_events_custom))
    splatted.register_hook(set_grad(splatted))
    splatted_custom.register_hook(set_grad(splatted_custom))
    loss.register_hook(set_grad(loss))
    loss_custom.register_hook(set_grad(loss_custom))

    loss_val = loss.sum()
    loss_val_custom = loss_custom.sum()
    print(f"Loss val: {loss_val.item()}\n")
    print(f"Loss val custom: {loss_val_custom.item()}\n")
    loss_val.backward()
    loss_val_custom.backward()

    print(f"Warped events gradients with shape {tuple(warped_events.grad.shape)}:\n{warped_events.grad}\n")
    print(f"Warped events gradients (custom) with shape {tuple(warped_events_custom.grad.shape)}:\n{warped_events_custom.grad}\n")
    visualize_tensor(flows.grad[..., 0], title="grad x flow")
    visualize_tensor(flows_custom.grad[..., 0], title="custom grad x flow")
    visualize_tensor(flows.grad[..., 1], title="grad y flow")
    visualize_tensor(flows_custom.grad[..., 1], title="custom grad y flow")
    visualize_tensor(splatted.grad, title="grad splatted image")
    visualize_tensor(splatted_custom.grad, title="custom grad splatted image")
    visualize_tensor(loss.grad, title="grad loss image")
    visualize_tensor(loss_custom.grad, title="custom grad loss image")
