import torch

from iterative_3d_warp_cuda import _C


class Iterative3DWarp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_fields, x_min, x_max, y_min, y_max):
        warped_points = _C.iterative_3d_warp_forward(points, flow_fields, x_min, x_max, y_min, y_max)
        ctx.save_for_backward(points, flow_fields, warped_points)
        ctx.x_min = x_min
        ctx.x_max = x_max
        ctx.y_min = y_min
        ctx.y_max = y_max
        return warped_points
    
    @staticmethod
    def backward(ctx, grad_output):
        points, flow_fields, warped_points = ctx.saved_tensors
        x_min = ctx.x_min
        x_max = ctx.x_max
        y_min = ctx.y_min
        y_max = ctx.y_max
        grad_points, grad_flow_fields = _C.iterative_3d_warp_backward(
            grad_output, points, flow_fields, warped_points, x_min, x_max, y_min, y_max
        )
        return grad_points, grad_flow_fields, None, None, None, None


def iterative_3d_warp(points, flow_fields, x_min, x_max, y_min, y_max):
    return Iterative3DWarp.apply(points, flow_fields, x_min, x_max, y_min, y_max)
