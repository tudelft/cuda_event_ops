import torch

from iterative_3d_warp_cuda._C import iterative_3d_warp_forward, iterative_3d_warp_backward
from trilinear_splat_cuda._C import trilinear_splat_forward, trilinear_splat_backward


class Iterative3DWarpCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_fields, num_warps, threads):
        warped_points = iterative_3d_warp_forward(points, flow_fields, num_warps)
        ctx.save_for_backward(points, flow_fields, warped_points)
        ctx.num_warps, ctx.threads = num_warps, threads
        return warped_points

    @staticmethod
    def backward(ctx, grad_output):
        points, flow_fields, warped_points = ctx.saved_tensors
        num_warps, threads = ctx.num_warps, ctx.threads
        grad_points, grad_flow_fields = iterative_3d_warp_backward(
            grad_output.contiguous(), points, flow_fields, warped_points, num_warps, threads
        )
        return grad_points, grad_flow_fields, None


def iterative_3d_warp_cuda(points, flow_fields, num_warps, threads=1024):
    return Iterative3DWarpCuda.apply(points, flow_fields, num_warps, threads)


class TrilinearSplatCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, grid_resolution, threads):
        splatted = trilinear_splat_forward(points, *grid_resolution)
        ctx.save_for_backward(points)
        ctx.grid_resolution, ctx.threads = grid_resolution, threads
        return splatted

    @staticmethod
    def backward(ctx, grad_output):
        (points,) = ctx.saved_tensors
        grid_resolution, threads = ctx.grid_resolution, ctx.threads
        grad_points = trilinear_splat_backward(grad_output.contiguous(), points, *grid_resolution, threads)
        return grad_points, None


def trilinear_splat_cuda(points, grid_resolution, threads=1024):
    return TrilinearSplatCuda.apply(points, grid_resolution, threads)
