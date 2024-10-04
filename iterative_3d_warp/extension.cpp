#include <torch/extension.h>


// cuda forward declarations
torch::Tensor iterative_3d_warp_cuda(
    torch::Tensor points,
    torch::Tensor flow_fields,
    int num_warps, bool keep_warping, int threads, int points_per_thread);


torch::Tensor iterative_3d_warp_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor flow_fields,
    torch::Tensor warped_points,
    int num_warps, int num_backprop_points, int threads, int points_per_thread);


// c++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor iterative_3d_warp_forward(
    torch::Tensor points,
    torch::Tensor flow_fields,
    int num_warps, bool keep_warping, int threads, int points_per_thread) {
  
    CHECK_INPUT(points);
    CHECK_INPUT(flow_fields);

    return iterative_3d_warp_cuda(points, flow_fields, num_warps, keep_warping, threads, points_per_thread);
}


torch::Tensor iterative_3d_warp_backward(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor flow_fields,
    torch::Tensor warped_points,
    int num_warps, int num_backprop_points, int threads, int points_per_thread) {
  
    CHECK_INPUT(grad_output);
    CHECK_INPUT(points);
    CHECK_INPUT(flow_fields);
    CHECK_INPUT(warped_points);

    return iterative_3d_warp_backward_cuda(grad_output, points, flow_fields, warped_points, num_warps, num_backprop_points, threads, points_per_thread);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iterative_3d_warp_forward", &iterative_3d_warp_forward, "Iterative 3D warp (CUDA)");
    m.def("iterative_3d_warp_backward", &iterative_3d_warp_backward, "Iterative 3D warp backward (CUDA)");
}
