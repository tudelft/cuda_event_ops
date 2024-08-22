#include <torch/extension.h>


// cuda forward declarations
torch::Tensor trilinear_splat_cuda(
    torch::Tensor points,
    torch::Tensor grid,
    int grid_d, int grid_h, int grid_w);


torch::Tensor trilinear_splat_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    int grid_d, int grid_h, int grid_w);


// c++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor trilinear_splat_forward(
    torch::Tensor points,
    int grid_d, int grid_h, int grid_w) {
  
    CHECK_INPUT(points);
    auto grid = torch::zeros({points.size(0), grid_d, grid_h, grid_w}, points.options());

    return trilinear_splat_cuda(points, grid, grid_d, grid_h, grid_w);
}


torch::Tensor trilinear_splat_backward(
    torch::Tensor grad_output,
    torch::Tensor points,
    int grid_d, int grid_h, int grid_w) {
  
    CHECK_INPUT(grad_output);
    CHECK_INPUT(points);

    return trilinear_splat_backward_cuda(grad_output, points, grid_d, grid_h, grid_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_splat_forward", &trilinear_splat_forward, "Trilinear splat (CUDA)");
    m.def("trilinear_splat_backward", &trilinear_splat_backward, "Trilinear splat backward (CUDA)");
}
