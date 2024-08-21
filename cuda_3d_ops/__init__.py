import torch

from cuda_3d_ops_cuda import _C


class Iterative3DWarpCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, flow_fields):
        warped_points = _C.iterative_3d_warp_forward(points, flow_fields)
        ctx.save_for_backward(points, flow_fields, warped_points)
        return warped_points
    
    @staticmethod
    def backward(ctx, grad_output):
        points, flow_fields, warped_points = ctx.saved_tensors
        grad_points, grad_flow_fields = _C.iterative_3d_warp_backward(grad_output, points, flow_fields, warped_points)
        return grad_points, grad_flow_fields


def iterative_3d_warp_cuda(points, flow_fields):
    return Iterative3DWarpCuda.apply(points, flow_fields)
