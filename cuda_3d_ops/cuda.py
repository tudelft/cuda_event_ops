import torch

from iterative_3d_warp_cuda._C import iterative_3d_warp_forward, iterative_3d_warp_backward
from trilinear_splat_cuda._C import trilinear_splat_forward, trilinear_splat_backward


class Iterative3DWarp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, events, flow_fields, num_warps, keep_warping, num_backprop_points, threads, points_per_thread):
        warped_events = iterative_3d_warp_forward(
            events, flow_fields, num_warps, keep_warping, threads, points_per_thread
        )
        ctx.save_for_backward(events, flow_fields, warped_events)
        ctx.num_warps, ctx.num_backprop_points, ctx.threads, ctx.points_per_thread = (
            num_warps,
            num_backprop_points,
            threads,
            points_per_thread,
        )
        return warped_events

    @staticmethod
    def backward(ctx, grad_output):
        events, flow_fields, warped_events = ctx.saved_tensors
        num_warps, num_backprop_points, threads, points_per_thread = (
            ctx.num_warps,
            ctx.num_backprop_points,
            ctx.threads,
            ctx.points_per_thread,
        )
        grad_flow_fields = iterative_3d_warp_backward(
            grad_output.contiguous(),
            events,
            flow_fields,
            warped_events,
            num_warps,
            num_backprop_points,
            threads,
            points_per_thread,
        )
        return None, grad_flow_fields, None, None, None, None, None


def iterative_3d_warp(
    events, flow_fields, num_warps, keep_warping=True, num_backprop_points=0, threads=1024, points_per_thread=1
):
    """
    Iteratively warps events in 3D using bilinearly-sampled flows.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, zi, val).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).
        num_warps (int): The number of warping stages.

    Returns:
        torch.Tensor: A tensor of shape (b, n, d + 1, 5), where each event has (x, y, z, z_orig, val).
    """
    return Iterative3DWarp.apply(
        events, flow_fields, num_warps, keep_warping, num_backprop_points, threads, points_per_thread
    )


class TrilinearSplat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, events, grid_resolution, threads, points_per_thread):
        splatted = trilinear_splat_forward(events, *grid_resolution, threads, points_per_thread)
        ctx.save_for_backward(events)
        ctx.grid_resolution, ctx.threads, ctx.points_per_thread = grid_resolution, threads, points_per_thread
        return splatted

    @staticmethod
    def backward(ctx, grad_output):
        (events,) = ctx.saved_tensors
        grid_resolution, threads, points_per_thread = ctx.grid_resolution, ctx.threads, ctx.points_per_thread
        grad_events = trilinear_splat_backward(
            grad_output.contiguous(), events, *grid_resolution, threads, points_per_thread
        )
        return grad_events, None, None, None


def trilinear_splat(events, grid_resolution, threads=1024, points_per_thread=1):
    """
    Trilinearly splats events into a 3D grid.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, z_orig, val).
        grid_resolution (tuple): The resolution of the output grid (d, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, d, h, w) with the splatted values.
    """
    return TrilinearSplat.apply(events, grid_resolution, threads, points_per_thread)
