import torch

from iterative_3d_warp_cuda._C import iterative_3d_warp_forward, iterative_3d_warp_backward
from trilinear_splat_cuda._C import trilinear_splat_forward, trilinear_splat_backward


class Iterative3DWarpCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_fields):
        warped_points = iterative_3d_warp_forward(points, flow_fields)
        ctx.save_for_backward(points, flow_fields, warped_points)
        return warped_points
    
    @staticmethod
    def backward(ctx, grad_output):
        points, flow_fields, warped_points = ctx.saved_tensors
        grad_points, grad_flow_fields = iterative_3d_warp_backward(grad_output.contiguous(), points, flow_fields, warped_points)
        return grad_points, grad_flow_fields


def iterative_3d_warp_cuda(points, flow_fields):
    return Iterative3DWarpCuda.apply(points, flow_fields)


class TrilinearSplatCuda(torch.autograd.Function):
    
        @staticmethod
        def forward(ctx, points, grid_resolution):
            splatted = trilinear_splat_forward(points, *grid_resolution)
            ctx.save_for_backward(points)
            ctx.grid_resolution = grid_resolution
            return splatted
        
        @staticmethod
        def backward(ctx, grad_output):
            (points,) = ctx.saved_tensors
            grid_resolution = ctx.grid_resolution
            grad_points = trilinear_splat_backward(grad_output.contiguous(), points, *grid_resolution)
            return grad_points, None


def trilinear_splat_cuda(points, grid_resolution):
    return TrilinearSplatCuda.apply(points, grid_resolution)
