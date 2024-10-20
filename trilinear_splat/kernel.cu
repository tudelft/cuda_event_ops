#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/extension.h>


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)


template <typename scalar_t>
__global__ void trilinear_splat_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid,
    int batch_size, int num_points, int grid_d, int grid_h, int grid_w) {
  
    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(index, batch_size * num_points) {
        // indices
        int bi = index / num_points;
        int pi = index % num_points;

        // get point
        scalar_t x = points[bi][pi][0];
        scalar_t y = points[bi][pi][1];
        scalar_t z = points[bi][pi][2];
        scalar_t val = points[bi][pi][3];

        // skip if value is zero
        if (val == 0) continue;
        
        // get corners and weights
        int x0 = floor(x), y0 = floor(y), z0 = floor(z);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

        scalar_t dx = x - x0, dy = y - y0, dz = z - z0;
        scalar_t wx0 = 1 - dx, wy0 = 1 - dy, wz0 = 1 - dz;
        scalar_t wx1 = dx, wy1 = dy, wz1 = dz;

        // zero weights give incorrect gradient perception for integer coordinates
        // wx0 = fmax(wx0, 1e-8);
        // wy0 = fmax(wy0, 1e-8);
        // wz0 = fmax(wz0, 1e-8);
        // wx1 = fmax(wx1, 1e-8);
        // wy1 = fmax(wy1, 1e-8);
        // wz1 = fmax(wz1, 1e-8);

        // compute conditions
        bool x0_in = x0 >= 0 && x0 < grid_w;
        bool x1_in = x1 >= 0 && x1 < grid_w;
        bool y0_in = y0 >= 0 && y0 < grid_h;
        bool y1_in = y1 >= 0 && y1 < grid_h;
        bool z0_in = z0 >= 0 && z0 < grid_d;
        bool z1_in = z1 >= 0 && z1 < grid_d;

        // add value if corner inside grid
        // pre-compute to be added value to save atomicAdd if zero
        scalar_t add_val;
        int numel = grid_d * grid_h * grid_w;
        if (x0_in && y0_in && z0_in) {
            add_val = val * wx0 * wy0 * wz0;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z0 * grid_h * grid_w + y0 * grid_w + x0, numel, add_val, true);
            }
        }
        if (x1_in && y0_in && z0_in) {
            add_val = val * wx1 * wy0 * wz0;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z0 * grid_h * grid_w + y0 * grid_w + x1, numel, add_val, true);
            }
        }
        if (x0_in && y1_in && z0_in) {
            add_val = val * wx0 * wy1 * wz0;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z0 * grid_h * grid_w + y1 * grid_w + x0, numel, add_val, true);
            }
        }
        if (x1_in && y1_in && z0_in) {
            add_val = val * wx1 * wy1 * wz0;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z0 * grid_h * grid_w + y1 * grid_w + x1, numel, add_val, true);
            }
        }
        if (x0_in && y0_in && z1_in) {
            add_val = val * wx0 * wy0 * wz1;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z1 * grid_h * grid_w + y0 * grid_w + x0, numel, add_val, true);
            }
        }
        if (x1_in && y0_in && z1_in) {
            add_val = val * wx1 * wy0 * wz1;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z1 * grid_h * grid_w + y0 * grid_w + x1, numel, add_val, true);
            }
        }
        if (x0_in && y1_in && z1_in) {
            add_val = val * wx0 * wy1 * wz1;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z1 * grid_h * grid_w + y1 * grid_w + x0, numel, add_val, true);
            }
        }
        if (x1_in && y1_in && z1_in) {
            add_val = val * wx1 * wy1 * wz1;
            if (add_val != 0.0f) {
                at::native::fastAtomicAdd(grid.data(), bi * numel + z1 * grid_h * grid_w + y1 * grid_w + x1, numel, add_val, true);
            }
        }
    }
}


template <typename scalar_t>
__global__ void trilinear_splat_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_points,
    int batch_size, int num_points, int grid_d, int grid_h, int grid_w) {
  
    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(index, batch_size * num_points) {
        // indices
        int bi = index / num_points;
        int pi = index % num_points;

        // get point
        scalar_t x = points[bi][pi][0];
        scalar_t y = points[bi][pi][1];
        scalar_t z = points[bi][pi][2];
        scalar_t val = points[bi][pi][3];

        // skip if value is zero
        if (val == 0) continue;

        // get corners and weights
        int x0 = floor(x), y0 = floor(y), z0 = floor(z);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

        scalar_t dx = x - x0, dy = y - y0, dz = z - z0;
        scalar_t wx0 = 1 - dx, wy0 = 1 - dy, wz0 = 1 - dz;
        scalar_t wx1 = dx, wy1 = dy, wz1 = dz;

        // compute conditions
        bool x0_in = x0 >= 0 && x0 < grid_w;
        bool x1_in = x1 >= 0 && x1 < grid_w;
        bool y0_in = y0 >= 0 && y0 < grid_h;
        bool y1_in = y1 >= 0 && y1 < grid_h;
        bool z0_in = z0 >= 0 && z0 < grid_d;
        bool z1_in = z1 >= 0 && z1 < grid_d;

        // add x and y gradients if corner was inside
        scalar_t grad_x = 0, grad_y = 0;
        if (x0_in && y0_in && z0_in) {
            grad_x -= grad_output[bi][z0][y0][x0] * wy0 * wz0 * val;
            grad_y -= grad_output[bi][z0][y0][x0] * wx0 * wz0 * val;
        }
        if (x1_in && y0_in && z0_in) {
            grad_x += grad_output[bi][z0][y0][x1] * wy0 * wz0 * val;
            grad_y -= grad_output[bi][z0][y0][x1] * wx1 * wz0 * val;
        }
        if (x0_in && y1_in && z0_in) {
            grad_x -= grad_output[bi][z0][y1][x0] * wy1 * wz0 * val;
            grad_y += grad_output[bi][z0][y1][x0] * wx0 * wz0 * val;
        }
        if (x1_in && y1_in && z0_in) {
            grad_x += grad_output[bi][z0][y1][x1] * wy1 * wz0 * val;
            grad_y += grad_output[bi][z0][y1][x1] * wx1 * wz0 * val;
        }
        if (x0_in && y0_in && z1_in) {
            grad_x -= grad_output[bi][z1][y0][x0] * wy0 * wz1 * val;
            grad_y -= grad_output[bi][z1][y0][x0] * wx0 * wz1 * val;
        }
        if (x1_in && y0_in && z1_in) {
            grad_x += grad_output[bi][z1][y0][x1] * wy0 * wz1 * val;
            grad_y -= grad_output[bi][z1][y0][x1] * wx1 * wz1 * val;
        }
        if (x0_in && y1_in && z1_in) {
            grad_x -= grad_output[bi][z1][y1][x0] * wy1 * wz1 * val;
            grad_y += grad_output[bi][z1][y1][x0] * wx0 * wz1 * val;
        }
        if (x1_in && y1_in && z1_in) {
            grad_x += grad_output[bi][z1][y1][x1] * wy1 * wz1 * val;
            grad_y += grad_output[bi][z1][y1][x1] * wx1 * wz1 * val;
        }

        // do atomic adds
        int numel = batch_size * num_points * 4;
        at::native::fastAtomicAdd(grad_points.data(), bi * num_points * 4 + pi * 4 + 0, numel, grad_x, true);
        at::native::fastAtomicAdd(grad_points.data(), bi * num_points * 4 + pi * 4 + 1, numel, grad_y, true);
    }
}


torch::Tensor trilinear_splat_cuda(
    torch::Tensor points,
    torch::Tensor grid,
    int grid_d, int grid_h, int grid_w, int threads, int points_per_thread) {
  
    int batch_size = points.size(0);
    int num_points = points.size(1);

    // one or multiple points per thread
    const int blocks = (batch_size * num_points + threads - 1) / threads / points_per_thread;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, points.scalar_type(), "trilinear_splat_cuda", [&] {
        trilinear_splat_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            grid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, num_points, grid_d, grid_h, grid_w);
    });

    return grid;
}


torch::Tensor trilinear_splat_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    int grid_d, int grid_h, int grid_w, int threads, int points_per_thread) {

    int batch_size = points.size(0);
    int num_points = points.size(1);

    auto grad_points = torch::zeros_like(points);

    // one or multiple points per thread
    const int blocks = (batch_size * num_points + threads - 1) / threads / points_per_thread;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, points.scalar_type(), "trilinear_splat_backward_cuda", [&] {
        trilinear_splat_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            grad_points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            batch_size, num_points, grid_d, grid_h, grid_w);
    });

    return grad_points;
}
