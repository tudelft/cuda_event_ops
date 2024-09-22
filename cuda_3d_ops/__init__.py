import torch

from iterative_3d_warp_cuda._C import iterative_3d_warp_forward, iterative_3d_warp_backward
from trilinear_splat_cuda._C import trilinear_splat_forward, trilinear_splat_backward


class Iterative3DWarpCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_fields, num_warps, keep_warping, threads, points_per_thread):
        warped_points = iterative_3d_warp_forward(points, flow_fields, num_warps, keep_warping, threads, points_per_thread)
        ctx.save_for_backward(points, flow_fields, warped_points)
        ctx.num_warps, ctx.threads, ctx.points_per_thread = num_warps, threads, points_per_thread
        return warped_points

    @staticmethod
    def backward(ctx, grad_output):
        points, flow_fields, warped_points = ctx.saved_tensors
        num_warps, threads, points_per_thread = ctx.num_warps, ctx.threads, ctx.points_per_thread
        grad_points, grad_flow_fields = iterative_3d_warp_backward(
            grad_output.contiguous(), points, flow_fields, warped_points, num_warps, threads, points_per_thread
        )
        return grad_points, grad_flow_fields, None, None, None, None


def iterative_3d_warp_cuda(points, flow_fields, num_warps, keep_warping, threads=1024, points_per_thread=1):
    return Iterative3DWarpCuda.apply(points, flow_fields, num_warps, keep_warping, threads, points_per_thread)


class TrilinearSplatCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, grid_resolution, threads, points_per_thread):
        splatted = trilinear_splat_forward(points, *grid_resolution, threads, points_per_thread)
        ctx.save_for_backward(points)
        ctx.grid_resolution, ctx.threads, ctx.points_per_thread = grid_resolution, threads, points_per_thread
        return splatted

    @staticmethod
    def backward(ctx, grad_output):
        (points,) = ctx.saved_tensors
        grid_resolution, threads, points_per_thread = ctx.grid_resolution, ctx.threads, ctx.points_per_thread
        grad_points = trilinear_splat_backward(grad_output.contiguous(), points, *grid_resolution, threads, points_per_thread)
        return grad_points, None, None, None


def trilinear_splat_cuda(points, grid_resolution, threads=1024, points_per_thread=1):
    return TrilinearSplatCuda.apply(points, grid_resolution, threads, points_per_thread)
