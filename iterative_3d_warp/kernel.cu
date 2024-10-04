#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/extension.h>


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)


template <typename scalar_t, typename index_t>
__device__ bool check_out_of_bounds(
    scalar_t x, scalar_t y,
    index_t x_min, index_t x_max, index_t y_min, index_t y_max) {
    return (x < static_cast<scalar_t>(x_min) || x >= static_cast<scalar_t>(x_max) || y < static_cast<scalar_t>(y_min) || y >= static_cast<scalar_t>(y_max));
}


template <typename scalar_t, typename index_t>
__device__ void bilinear_interpolation(
    const scalar_t* flow_field,
    index_t height, index_t width, scalar_t x, scalar_t y,
    scalar_t& flow_x, scalar_t& flow_y) {

    index_t x0 = floor(x);
    index_t y0 = floor(y);
    index_t x1 = x0 + 1;
    index_t y1 = y0 + 1;

    scalar_t w00 = (x1 - x) * (y1 - y);  // top left
    scalar_t w01 = (x - x0) * (y1 - y);  // top right
    scalar_t w10 = (x1 - x) * (y - y0);  // bottom left
    scalar_t w11 = (x - x0) * (y - y0);  // bottom right

    flow_x = w00 * flow_field[(y0 * width + x0) * 2] +
             w01 * flow_field[(y0 * width + x1) * 2] +
             w10 * flow_field[(y1 * width + x0) * 2] +
             w11 * flow_field[(y1 * width + x1) * 2];

    flow_y = w00 * flow_field[(y0 * width + x0) * 2 + 1] +
             w01 * flow_field[(y0 * width + x1) * 2 + 1] +
             w10 * flow_field[(y1 * width + x0) * 2 + 1] +
             w11 * flow_field[(y1 * width + x1) * 2 + 1];
}


template <typename scalar_t>
__global__ void iterative_3d_warp_kernel(
    const scalar_t* __restrict__ points,
    const scalar_t* __restrict__ flow_fields,
    scalar_t* __restrict__ warped_points,
    int batch_size, int num_points, int num_flow_fields, int num_z, int num_warps, bool keep_warping, int height, int width) {
    
    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(idx, batch_size * num_points) {
        // indices
        int batch_idx = idx / num_points;
        int point_idx = idx % num_points;

        // load point
        scalar_t x = points[batch_idx * num_points * 5 + point_idx * 5];
        scalar_t y = points[batch_idx * num_points * 5 + point_idx * 5 + 1];
        scalar_t z = points[batch_idx * num_points * 5 + point_idx * 5 + 2];
        int zi = points[batch_idx * num_points * 5 + point_idx * 5 + 3];
        scalar_t val = points[batch_idx * num_points * 5 + point_idx * 5 + 4];
        scalar_t z_orig = z;

        // skip if value is zero
        if (val == 0) return;

        // if out of bounds to start with, return
        bool is_out_of_bounds = check_out_of_bounds(x, y, 0, width - 1, 0, height - 1);
        if (is_out_of_bounds) return;

        // warp forward: increasing z values
        // start with next integer z value
        for (int z_next = zi + 1; z_next < num_z; z_next++) {
            scalar_t dz = z_next - z;

            // if keep_warping: value to 0 if number of warps reached; else break
            if (z_next - zi > num_warps) {
                if (keep_warping) {
                    val = 0;
                } else {
                    break;
                }
            }

            // bilinear interpolation to get flow at (x, y)
            const scalar_t* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2;
            scalar_t flow_x, flow_y;
            bilinear_interpolation(flow_field, height, width, x, y, flow_x, flow_y);
            
            // warp point position
            // scale flow by dz
            x += flow_x * dz;
            y += flow_y * dz;
            z = z_next;  // prevents rounding error?; same as z += dz

            // save warped point position
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
            warped_points[output_idx] = x;
            warped_points[output_idx + 1] = y;
            warped_points[output_idx + 2] = z;
            warped_points[output_idx + 3] = z_orig;
            warped_points[output_idx + 4] = val;

            // check bounds
            if (check_out_of_bounds(x, y, 0, width - 1, 0, height - 1)) {
                is_out_of_bounds = true;
                break;  // stop updating this point if it goes out of bounds
            }
        }

        // only do if not out of bounds
        if (!is_out_of_bounds) {
            // reload point
            x = points[batch_idx * num_points * 5 + point_idx * 5];
            y = points[batch_idx * num_points * 5 + point_idx * 5 + 1];
            z = points[batch_idx * num_points * 5 + point_idx * 5 + 2];
            zi = points[batch_idx * num_points * 5 + point_idx * 5 + 3];
            val = points[batch_idx * num_points * 5 + point_idx * 5 + 4];

            // warp backward: decreasing z values
            // start with previous integer z value
            for (int z_next = zi; z_next > -1; z_next--) {
                scalar_t dz = z - z_next;

                // if keep_warping: value to 0 if max number of warps reached, else break
                // using floored index so -1
                if (zi - z_next > num_warps - 1) {
                    if (keep_warping) {
                        val = 0;
                    } else {
                        break;
                    }
                }

                // bilinear interpolation to get flow at (x, y)
                const scalar_t* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2;
                scalar_t flow_x, flow_y;
                bilinear_interpolation(flow_field, height, width, x, y, flow_x, flow_y);

                // warp point position
                // scale flow by dz
                x -= flow_x * dz;
                y -= flow_y * dz;
                z = z_next;  // same as z -= dz

                // save warped point position
                int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
                warped_points[output_idx] = x;
                warped_points[output_idx + 1] = y;
                warped_points[output_idx + 2] = z;
                warped_points[output_idx + 3] = z_orig;
                warped_points[output_idx + 4] = val;

                // check bounds
                if (check_out_of_bounds(x, y, 0, width - 1, 0, height - 1)) {
                    is_out_of_bounds = true;
                    break;  // stop updating this point if it goes out of bounds
                }
            }
        }

        // set all values to zero if out of bounds at some point
        if (is_out_of_bounds) {
            for (int z = 0; z < num_z; z++) {
                int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z * 5;
                warped_points[output_idx + 4] = 0;
            }
        }
    }
}


template <typename scalar_t>
__global__ void iterative_3d_warp_backward_kernel(
    const scalar_t* __restrict__ grad_output, 
    const scalar_t* __restrict__ points, 
    const scalar_t* __restrict__ flow_fields,
    const scalar_t* __restrict__ warped_points,
    const bool* __restrict__ backprop_point,
    scalar_t* __restrict__ grad_flow_fields,
    int batch_size, int num_points, int num_flow_fields, int num_z, int num_warps, int height, int width) {

    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(idx, batch_size * num_points) {
        // indices
        int batch_idx = idx / num_points;
        int point_idx = idx % num_points;

        // only backprop fraction/number of points
        if (!backprop_point[batch_idx * num_points + point_idx]) return;

        // load starting z
        scalar_t z_orig = points[batch_idx * num_points * 5 + point_idx * 5 + 2];
        int zi = points[batch_idx * num_points * 5 + point_idx * 5 + 3];

        // check if point was out of bounds (or zero padding): all values are zero
        if (warped_points[batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + zi * 5 + 4] == 0) return;

        // accumulate gradients for points
        scalar_t grad_warped_point_x = 0;
        scalar_t grad_warped_point_y = 0;

        // iterate over z values in reverse for forward warping gradient computation
        for (int z_next = min(num_z - 1, zi + num_warps); z_next > zi; z_next--) {
            // get previous warped point position
            scalar_t prev_x, prev_y, dz;
            if (z_next == zi + 1) {
                prev_x = points[batch_idx * num_points * 5 + point_idx * 5];
                prev_y = points[batch_idx * num_points * 5 + point_idx * 5 + 1];
                dz = z_next - z_orig;
            } else {
                int prev_output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + (z_next - 1) * 5;
                prev_x = warped_points[prev_output_idx];
                prev_y = warped_points[prev_output_idx + 1];
                dz = 1;
            }

            // get bilinear interpolation weights
            int x0 = floor(prev_x);
            int y0 = floor(prev_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            scalar_t w00 = (x1 - prev_x) * (y1 - prev_y);  // top left
            scalar_t w01 = (prev_x - x0) * (y1 - prev_y);  // top right
            scalar_t w10 = (x1 - prev_x) * (prev_y - y0);  // bottom left
            scalar_t w11 = (prev_x - x0) * (prev_y - y0);  // bottom right

            // add output gradients
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
            grad_warped_point_x += grad_output[output_idx];
            grad_warped_point_y += grad_output[output_idx + 1];

            // pre-compute to be added value to avoid expensive atomicAdd if zero
            scalar_t val_x00 = grad_warped_point_x * w00 * dz;
            scalar_t val_x01 = grad_warped_point_x * w01 * dz;
            scalar_t val_x10 = grad_warped_point_x * w10 * dz;
            scalar_t val_x11 = grad_warped_point_x * w11 * dz;
            scalar_t val_y00 = grad_warped_point_y * w00 * dz;
            scalar_t val_y01 = grad_warped_point_y * w01 * dz;
            scalar_t val_y10 = grad_warped_point_y * w10 * dz;
            scalar_t val_y11 = grad_warped_point_y * w11 * dz;

            // add grads wrt x flow
            int numel = batch_size * num_flow_fields * height * width * 2;
            if (val_x00 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2, numel, val_x00, true);
            }
            if (val_x01 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2, numel, val_x01, true);
            }
            if (val_x10 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2, numel, val_x10, true);
            }
            if (val_x11 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2, numel, val_x11, true);
            }
            
            // add grads wrt y flow
            if (val_y00 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2 + 1, numel, val_y00, true);
            }
            if (val_y01 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2 + 1, numel, val_y01, true);
            }
            if (val_y10 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2 + 1, numel, val_y10, true);
            }
            if (val_y11 != 0.0f) {
                at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2 + 1, numel, val_y11, true);
            }

            // calculate grad wrt xy
            // changes in flow field
            int idx_00 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2;
            int idx_01 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2;
            int idx_10 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2;
            int idx_11 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2;

            scalar_t f00_x = flow_fields[idx_00];
            scalar_t f01_x = flow_fields[idx_01];
            scalar_t f10_x = flow_fields[idx_10];
            scalar_t f11_x = flow_fields[idx_11];
            scalar_t f00_y = flow_fields[idx_00 + 1];
            scalar_t f01_y = flow_fields[idx_01 + 1];
            scalar_t f10_y = flow_fields[idx_10 + 1];
            scalar_t f11_y = flow_fields[idx_11 + 1];

            scalar_t dflowx_dx = (f01_x - f00_x) * (y1 - prev_y) + (f11_x - f10_x) * (prev_y - y0);
            scalar_t dflowy_dx = (f01_y - f00_y) * (y1 - prev_y) + (f11_y - f10_y) * (prev_y - y0);
            scalar_t dflowx_dy = (f10_x - f00_x) * (x1 - prev_x) + (f11_x - f01_x) * (prev_x - x0);
            scalar_t dflowy_dy = (f10_y - f00_y) * (x1 - prev_x) + (f11_y - f01_y) * (prev_x - x0);

            // add grads wrt x and y point
            // TODO: this looks wrong, but is correct?
            scalar_t grad_point_x = grad_warped_point_x * (1 + dflowx_dx * dz) + grad_warped_point_y * dflowy_dx * dz;
            scalar_t grad_point_y = grad_warped_point_x * dflowx_dy * dz + grad_warped_point_y * (1 + dflowy_dy * dz);
            grad_warped_point_x = grad_point_x;
            grad_warped_point_y = grad_point_y;
        }

        // reset gradients
        grad_warped_point_x = 0;
        grad_warped_point_y = 0;

        // iterate over z values in reverse for backward warping gradient computation
        for (int z_next = max(0, zi - (num_warps - 1)); z_next <= zi; z_next++) {
            // get previous warped point position
            scalar_t prev_x, prev_y, dz;
            if (z_next == zi) {
                prev_x = points[batch_idx * num_points * 5 + point_idx * 5];
                prev_y = points[batch_idx * num_points * 5 + point_idx * 5 + 1];
                dz = z_orig - z_next;
            } else {
                int prev_output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + (z_next + 1) * 5;
                prev_x = warped_points[prev_output_idx];
                prev_y = warped_points[prev_output_idx + 1];
                dz = 1;
            }

            // get bilinear interpolation weights
            int x0 = floor(prev_x);
            int y0 = floor(prev_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            scalar_t w00 = (x1 - prev_x) * (y1 - prev_y);  // top left
            scalar_t w01 = (prev_x - x0) * (y1 - prev_y);  // top right
            scalar_t w10 = (x1 - prev_x) * (prev_y - y0);  // bottom left
            scalar_t w11 = (prev_x - x0) * (prev_y - y0);  // bottom right

            // add output gradients
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
            grad_warped_point_x += grad_output[output_idx];
            grad_warped_point_y += grad_output[output_idx + 1];

            // add grads wrt x flow
            int numel = batch_size * num_flow_fields * height * width * 2;
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2, numel, grad_warped_point_x * w00 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2, numel, grad_warped_point_x * w01 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2, numel, grad_warped_point_x * w10 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2, numel, grad_warped_point_x * w11 * -dz, true);
            
            // add grads wrt y flow
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2 + 1, numel, grad_warped_point_y * w00 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2 + 1, numel, grad_warped_point_y * w01 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2 + 1, numel, grad_warped_point_y * w10 * -dz, true);
            at::native::fastAtomicAdd(grad_flow_fields, batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2 + 1, numel, grad_warped_point_y * w11 * -dz, true);

            // calculate grad wrt xy
            // changes in flow field
            int idx_00 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2;
            int idx_01 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2;
            int idx_10 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2;
            int idx_11 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2;

            scalar_t f00_x = flow_fields[idx_00];
            scalar_t f01_x = flow_fields[idx_01];
            scalar_t f10_x = flow_fields[idx_10];
            scalar_t f11_x = flow_fields[idx_11];
            scalar_t f00_y = flow_fields[idx_00 + 1];
            scalar_t f01_y = flow_fields[idx_01 + 1];
            scalar_t f10_y = flow_fields[idx_10 + 1];
            scalar_t f11_y = flow_fields[idx_11 + 1];

            scalar_t dflowx_dx = (f01_x - f00_x) * (y1 - prev_y) + (f11_x - f10_x) * (prev_y - y0);
            scalar_t dflowy_dx = (f01_y - f00_y) * (y1 - prev_y) + (f11_y - f10_y) * (prev_y - y0);
            scalar_t dflowx_dy = (f10_x - f00_x) * (x1 - prev_x) + (f11_x - f01_x) * (prev_x - x0);
            scalar_t dflowy_dy = (f10_y - f00_y) * (x1 - prev_x) + (f11_y - f01_y) * (prev_x - x0);

            // add grads wrt x and y point
            // TODO: this looks wrong, but is correct?
            scalar_t grad_point_x = grad_warped_point_x * (1 + dflowx_dx * -dz) + grad_warped_point_y * dflowy_dx * -dz;
            scalar_t grad_point_y = grad_warped_point_x * dflowx_dy * -dz + grad_warped_point_y * (1 + dflowy_dy * -dz);
            grad_warped_point_x = grad_point_x;
            grad_warped_point_y = grad_point_y;
        }
    }
}


torch::Tensor iterative_3d_warp_cuda(
    torch::Tensor points,
    torch::Tensor flow_fields,
    int num_warps, bool keep_warping, int threads, int points_per_thread) {

    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_flow_fields = flow_fields.size(1);
    int num_z = num_flow_fields + 1;
    int height = flow_fields.size(2);
    int width = flow_fields.size(3);

    // points: (b, n, 5)
    // flow_fields: (b, d, h, w, 2)
    // warped_points: (b, n, d + 1, 5)
    auto warped_points = torch::zeros({batch_size, num_points, num_z, 5}, points.options());

    // one or multiple points per thread
    int blocks = (batch_size * num_points + threads - 1) / threads / points_per_thread;

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, points.scalar_type(), "iterative_3d_warp_cuda", [&] {
        iterative_3d_warp_kernel<scalar_t><<<blocks, threads>>>(
            points.data_ptr<scalar_t>(),
            flow_fields.data_ptr<scalar_t>(),
            warped_points.data_ptr<scalar_t>(),
            batch_size, num_points, num_flow_fields, num_z, num_warps, keep_warping, height, width);
    });

    return warped_points;
}


torch::Tensor iterative_3d_warp_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor flow_fields,
    torch::Tensor warped_points,
    int num_warps, int num_backprop_points, int threads, int points_per_thread) {

    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_flow_fields = flow_fields.size(1);
    int num_z = num_flow_fields + 1;
    int height = flow_fields.size(2);
    int width = flow_fields.size(3);

    // generate bool tensor with set number of points to backprop per batch
    // TODO: this might include zero padding for grads (useless)
    auto backprop_point = torch::zeros_like(points, torch::kBool);
    if (num_backprop_points > 0) {
        num_backprop_points = min(num_backprop_points, num_points);
        auto sample = torch::rand_like(points);
        auto topk_result = torch::topk(sample, num_backprop_points, /*dim=*/1);
        auto top_indices = std::get<1>(topk_result);
        backprop_point.scatter_(1, top_indices, true);
    } else {
        backprop_point.fill_(true);
    }

    auto grad_flow_fields = torch::zeros_like(flow_fields);

    // one or multiple points per thread
    int blocks = (batch_size * num_points + threads - 1) / threads / points_per_thread;

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, points.scalar_type(), "iterative_3d_warp_backward_cuda", [&] {
        iterative_3d_warp_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            points.data_ptr<scalar_t>(),
            flow_fields.data_ptr<scalar_t>(),
            warped_points.data_ptr<scalar_t>(),
            backprop_point.data_ptr<bool>(),
            grad_flow_fields.data_ptr<scalar_t>(),
            batch_size, num_points, num_flow_fields, num_z, num_warps, height, width);
    });

    return grad_flow_fields;
}
