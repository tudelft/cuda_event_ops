#include <torch/extension.h>


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)


__global__ void trilinear_splat_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grid,
    int batch_size, int num_points, int grid_d, int grid_h, int grid_w) {
  
    // one thread can handle multiple points
    // useful when points >> threads
    // TODO: compare performance with one point per thread
    CUDA_KERNEL_LOOP(index, batch_size * num_points) {
        // indices
        int bi = index / num_points;
        int pi = index % num_points;

        // get point
        float x = points[bi][pi][0];
        float y = points[bi][pi][1];
        float z = points[bi][pi][2];
        float val = points[bi][pi][3];
        
        // get corners and weights
        int x0 = floor(x), y0 = floor(y), z0 = floor(z);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

        float dx = x - x0, dy = y - y0, dz = z - z0;
        float wx0 = 1 - dx, wy0 = 1 - dy, wz0 = 1 - dz;
        float wx1 = dx, wy1 = dy, wz1 = dz;

        // add value if corner inside grid
        if (x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 < grid_w && y0 < grid_h && z0 < grid_d) {
            atomicAdd(&grid[bi][z0][y0][x0], val * wx0 * wy0 * wz0);
        }
        if (x1 >= 0 && y0 >= 0 && z0 >= 0 && x1 < grid_w && y0 < grid_h && z0 < grid_d) {
            atomicAdd(&grid[bi][z0][y0][x1], val * wx1 * wy0 * wz0);
        }
        if (x0 >= 0 && y1 >= 0 && z0 >= 0 && x0 < grid_w && y1 < grid_h && z0 < grid_d) {
            atomicAdd(&grid[bi][z0][y1][x0], val * wx0 * wy1 * wz0);
        }
        if (x1 >= 0 && y1 >= 0 && z0 >= 0 && x1 < grid_w && y1 < grid_h && z0 < grid_d) {
            atomicAdd(&grid[bi][z0][y1][x1], val * wx1 * wy1 * wz0);
        }
        if (x0 >= 0 && y0 >= 0 && z1 >= 0 && x0 < grid_w && y0 < grid_h && z1 < grid_d) {
            atomicAdd(&grid[bi][z1][y0][x0], val * wx0 * wy0 * wz1);
        }
        if (x1 >= 0 && y0 >= 0 && z1 >= 0 && x1 < grid_w && y0 < grid_h && z1 < grid_d) {
            atomicAdd(&grid[bi][z1][y0][x1], val * wx1 * wy0 * wz1);
        }
        if (x0 >= 0 && y1 >= 0 && z1 >= 0 && x0 < grid_w && y1 < grid_h && z1 < grid_d) {
            atomicAdd(&grid[bi][z1][y1][x0], val * wx0 * wy1 * wz1);
        }
        if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x1 < grid_w && y1 < grid_h && z1 < grid_d) {
            atomicAdd(&grid[bi][z1][y1][x1], val * wx1 * wy1 * wz1);
        }
    }
}


__global__ void trilinear_splat_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_points,
    int batch_size, int num_points, int grid_d, int grid_h, int grid_w) {
  
    // one thread can handle multiple points
    // useful when points >> threads
    // TODO: compare performance with one point per thread
    CUDA_KERNEL_LOOP(index, batch_size * num_points) {
        // indices
        int bi = index / num_points;
        int pi = index % num_points;

        // get point
        float x = points[bi][pi][0];
        float y = points[bi][pi][1];
        float z = points[bi][pi][2];
        float val = points[bi][pi][3];
        
        // get corners and weights
        int x0 = floor(x), y0 = floor(y), z0 = floor(z);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

        float dx = x - x0, dy = y - y0, dz = z - z0;
        float wx0 = 1 - dx, wy0 = 1 - dy, wz0 = 1 - dz;
        float wx1 = dx, wy1 = dy, wz1 = dz;

        // add x and y gradients if corner was inside
        // TODO: better to have atomicAdds for every if or one at the end?
        if (x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 < grid_w && y0 < grid_h && z0 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], -grad_output[bi][z0][y0][x0] * wy0 * wz0 * val);
            atomicAdd(&grad_points[bi][pi][1], -grad_output[bi][z0][y0][x0] * wx0 * wz0 * val);
        }
        if (x1 >= 0 && y0 >= 0 && z0 >= 0 && x1 < grid_w && y0 < grid_h && z0 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], grad_output[bi][z0][y0][x1] * wy0 * wz0 * val);
            atomicAdd(&grad_points[bi][pi][1], -grad_output[bi][z0][y0][x1] * wx1 * wz0 * val);
        }
        if (x0 >= 0 && y1 >= 0 && z0 >= 0 && x0 < grid_w && y1 < grid_h && z0 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], -grad_output[bi][z0][y1][x0] * wy1 * wz0 * val);
            atomicAdd(&grad_points[bi][pi][1], grad_output[bi][z0][y1][x0] * wx0 * wz0 * val);
        }
        if (x1 >= 0 && y1 >= 0 && z0 >= 0 && x1 < grid_w && y1 < grid_h && z0 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], grad_output[bi][z0][y1][x1] * wy1 * wz0 * val);
            atomicAdd(&grad_points[bi][pi][1], grad_output[bi][z0][y1][x1] * wx1 * wz0 * val);
        }
        if (x0 >= 0 && y0 >= 0 && z1 >= 0 && x0 < grid_w && y0 < grid_h && z1 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], -grad_output[bi][z1][y0][x0] * wy0 * wz1 * val);
            atomicAdd(&grad_points[bi][pi][1], -grad_output[bi][z1][y0][x0] * wx0 * wz1 * val);
        }
        if (x1 >= 0 && y0 >= 0 && z1 >= 0 && x1 < grid_w && y0 < grid_h && z1 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], grad_output[bi][z1][y0][x1] * wy0 * wz1 * val);
            atomicAdd(&grad_points[bi][pi][1], -grad_output[bi][z1][y0][x1] * wx1 * wz1 * val);
        }
        if (x0 >= 0 && y1 >= 0 && z1 >= 0 && x0 < grid_w && y1 < grid_h && z1 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], -grad_output[bi][z1][y1][x0] * wy1 * wz1 * val);
            atomicAdd(&grad_points[bi][pi][1], grad_output[bi][z1][y1][x0] * wx0 * wz1 * val);
        }
        if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x1 < grid_w && y1 < grid_h && z1 < grid_d) {
            atomicAdd(&grad_points[bi][pi][0], grad_output[bi][z1][y1][x1] * wy1 * wz1 * val);
            atomicAdd(&grad_points[bi][pi][1], grad_output[bi][z1][y1][x1] * wx1 * wz1 * val);
        }
    }
}


torch::Tensor trilinear_splat_cuda(
    torch::Tensor points,
    torch::Tensor grid,
    int grid_d, int grid_h, int grid_w) {
  
    int batch_size = points.size(0);
    int num_points = points.size(1);

    // only if at least one point
    printf("num_points: %d\n", num_points);
    if (num_points == 0) return grid;

    // one thread per point
    // TODO: less is more optimal?
    const int threads = 1024;
    const int blocks = (batch_size * num_points + threads - 1) / threads;
    
    trilinear_splat_kernel<<<blocks, threads>>>(
        points.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        grid.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        batch_size, num_points, grid_d, grid_h, grid_w);

    return grid;
}


torch::Tensor trilinear_splat_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    int grid_d, int grid_h, int grid_w) {

    int batch_size = points.size(0);
    int num_points = points.size(1);

    auto grad_points = torch::zeros_like(points);

    // only if at least one point
    printf("num_points: %d\n", num_points);
    if (num_points == 0) return grad_points;
    
    // one thread per point
    // TODO: less is more optimal?
    const int threads = 1024;
    const int blocks = (batch_size * num_points + threads - 1) / threads;
    
    trilinear_splat_backward_kernel<<<blocks, threads>>>(
        grad_output.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        points.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        grad_points.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        batch_size, num_points, grid_d, grid_h, grid_w);

    return grad_points;
}
