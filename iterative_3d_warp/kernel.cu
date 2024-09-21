#include <torch/extension.h>
#include <vector>


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)


__device__ bool check_out_of_bounds(
    float x, float y,
    float x_min, float x_max, float y_min, float y_max) {
    return (x < x_min || x >= x_max || y < y_min || y >= y_max);
}


__device__ void bilinear_interpolation(
    const float* flow_field,
    int height, int width, float x, float y,
    float& flow_x, float& flow_y) {

    int x0 = floor(x);
    int y0 = floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float w00 = (x1 - x) * (y1 - y);  // top left
    float w01 = (x - x0) * (y1 - y);  // top right
    float w10 = (x1 - x) * (y - y0);  // bottom left
    float w11 = (x - x0) * (y - y0);  // bottom right

    flow_x = w00 * flow_field[(y0 * width + x0) * 2] +
             w01 * flow_field[(y0 * width + x1) * 2] +
             w10 * flow_field[(y1 * width + x0) * 2] +
             w11 * flow_field[(y1 * width + x1) * 2];

    flow_y = w00 * flow_field[(y0 * width + x0) * 2 + 1] +
             w01 * flow_field[(y0 * width + x1) * 2 + 1] +
             w10 * flow_field[(y1 * width + x0) * 2 + 1] +
             w11 * flow_field[(y1 * width + x1) * 2 + 1];
}


__global__ void iterative_3d_warp_kernel(
    const float* __restrict__ points, 
    const float* __restrict__ flow_fields, 
    float* __restrict__ warped_points,
    int batch_size, int num_points, int num_flow_fields, int num_z, int num_warps, int height, int width) {
    
    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(idx, batch_size * num_points) {
        // indices
        int batch_idx = idx / num_points;
        int point_idx = idx % num_points;

        // load point
        float x = points[batch_idx * num_points * 5 + point_idx * 5];
        float y = points[batch_idx * num_points * 5 + point_idx * 5 + 1];
        float z = points[batch_idx * num_points * 5 + point_idx * 5 + 2];
        int zi = points[batch_idx * num_points * 5 + point_idx * 5 + 3];
        float val = points[batch_idx * num_points * 5 + point_idx * 5 + 4];
        float z_orig = z;

        // skip if value is zero
        if (val == 0) return;

        // if out of bounds to start with, return
        bool is_out_of_bounds = check_out_of_bounds(x, y, 0, width - 1, 0, height - 1);
        if (is_out_of_bounds) return;

        // warp forward: increasing z values
        // start with next integer z value
        for (int z_next = zi + 1; z_next < num_z; z_next++) {
            float dz = z_next - z;

            // value to 0 if number of warps reached
            if (z_next - zi > num_warps) val = 0;

            // bilinear interpolation to get flow at (x, y)
            const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2;
            float flow_x, flow_y;
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
                float dz = z - z_next;

                // value to 0 if max number of warps reached
                // using floored index so -1
                if (zi - z_next > num_warps - 1) val = 0;

                // bilinear interpolation to get flow at (x, y)
                const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2;
                float flow_x, flow_y;
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


__global__ void iterative_3d_warp_backward_kernel(
    const float* __restrict__ grad_output, 
    const float* __restrict__ points, 
    const float* __restrict__ flow_fields,
    const float* __restrict__ warped_points,
    float* __restrict__ grad_points,
    float* __restrict__ grad_flow_fields,
    int batch_size, int num_points, int num_flow_fields, int num_z, int num_warps, int height, int width) {

    // one thread can handle multiple points
    // useful when points >> threads
    CUDA_KERNEL_LOOP(idx, batch_size * num_points) {
        // indices
        int batch_idx = idx / num_points;
        int point_idx = idx % num_points;

        // load starting z
        float z_orig = points[batch_idx * num_points * 5 + point_idx * 5 + 2];
        int zi = points[batch_idx * num_points * 5 + point_idx * 5 + 3];

        // check if point was out of bounds (or zero padding): all values are zero
        if (warped_points[batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + zi * 5 + 4] == 0) return;

        // accumulate gradients for points
        float grad_warped_point_x = 0;
        float grad_warped_point_y = 0;

        // iterate over z values in reverse for forward warping gradient computation
        for (int z_next = min(num_z - 1, zi + num_warps); z_next > zi; z_next--) {
            // get previous warped point position
            float prev_x, prev_y, dz;
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

            float w00 = (x1 - prev_x) * (y1 - prev_y);  // top left
            float w01 = (prev_x - x0) * (y1 - prev_y);  // top right
            float w10 = (x1 - prev_x) * (prev_y - y0);  // bottom left
            float w11 = (prev_x - x0) * (prev_y - y0);  // bottom right

            // add output gradients
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
            grad_warped_point_x += grad_output[output_idx];
            grad_warped_point_y += grad_output[output_idx + 1];

            // add grads wrt x flow
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2], grad_warped_point_x * w00 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2], grad_warped_point_x * w01 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2], grad_warped_point_x * w10 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2], grad_warped_point_x * w11 * dz);
            
            // add grads wrt y flow
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2 + 1], grad_warped_point_y * w00 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2 + 1], grad_warped_point_y * w01 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2 + 1], grad_warped_point_y * w10 * dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2 + 1], grad_warped_point_y * w11 * dz);

            // calculate grad wrt xy
            // changes in flow field
            int idx_00 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x0) * 2;
            int idx_01 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y0 * width + x1) * 2;
            int idx_10 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x0) * 2;
            int idx_11 = batch_idx * num_flow_fields * height * width * 2 + (z_next - 1) * height * width * 2 + (y1 * width + x1) * 2;

            float f00_x = flow_fields[idx_00];
            float f01_x = flow_fields[idx_01];
            float f10_x = flow_fields[idx_10];
            float f11_x = flow_fields[idx_11];
            float f00_y = flow_fields[idx_00 + 1];
            float f01_y = flow_fields[idx_01 + 1];
            float f10_y = flow_fields[idx_10 + 1];
            float f11_y = flow_fields[idx_11 + 1];

            float dflowx_dx = (f01_x - f00_x) * (y1 - prev_y) + (f11_x - f10_x) * (prev_y - y0);
            float dflowy_dx = (f01_y - f00_y) * (y1 - prev_y) + (f11_y - f10_y) * (prev_y - y0);
            float dflowx_dy = (f10_x - f00_x) * (x1 - prev_x) + (f11_x - f01_x) * (prev_x - x0);
            float dflowy_dy = (f10_y - f00_y) * (x1 - prev_x) + (f11_y - f01_y) * (prev_x - x0);

            // add grads wrt x and y point
            // TODO: this looks wrong, but is correct?
            float grad_point_x = grad_warped_point_x * (1 + dflowx_dx * dz) + grad_warped_point_y * dflowy_dx * dz;
            float grad_point_y = grad_warped_point_x * dflowx_dy * dz + grad_warped_point_y * (1 + dflowy_dy * dz);
            grad_warped_point_x = grad_point_x;
            grad_warped_point_y = grad_point_y;
        }

        // reset gradients
        grad_warped_point_x = 0;
        grad_warped_point_y = 0;

        // iterate over z values in reverse for backward warping gradient computation
        for (int z_next = max(0, zi - (num_warps - 1)); z_next <= zi; z_next++) {
            // get previous warped point position
            float prev_x, prev_y, dz;
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

            float w00 = (x1 - prev_x) * (y1 - prev_y);  // top left
            float w01 = (prev_x - x0) * (y1 - prev_y);  // top right
            float w10 = (x1 - prev_x) * (prev_y - y0);  // bottom left
            float w11 = (prev_x - x0) * (prev_y - y0);  // bottom right

            // add output gradients
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z_next * 5;
            grad_warped_point_x += grad_output[output_idx];
            grad_warped_point_y += grad_output[output_idx + 1];

            // add grads wrt x flow
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2], grad_warped_point_x * w00 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2], grad_warped_point_x * w01 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2], grad_warped_point_x * w10 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2], grad_warped_point_x * w11 * -dz);
            
            // add grads wrt y flow
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2 + 1], grad_warped_point_y * w00 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2 + 1], grad_warped_point_y * w01 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2 + 1], grad_warped_point_y * w10 * -dz);
            atomicAdd(&grad_flow_fields[batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2 + 1], grad_warped_point_y * w11 * -dz);

            // calculate grad wrt xy
            // changes in flow field
            int idx_00 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x0) * 2;
            int idx_01 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y0 * width + x1) * 2;
            int idx_10 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x0) * 2;
            int idx_11 = batch_idx * num_flow_fields * height * width * 2 + z_next * height * width * 2 + (y1 * width + x1) * 2;

            float f00_x = flow_fields[idx_00];
            float f01_x = flow_fields[idx_01];
            float f10_x = flow_fields[idx_10];
            float f11_x = flow_fields[idx_11];
            float f00_y = flow_fields[idx_00 + 1];
            float f01_y = flow_fields[idx_01 + 1];
            float f10_y = flow_fields[idx_10 + 1];
            float f11_y = flow_fields[idx_11 + 1];

            float dflowx_dx = (f01_x - f00_x) * (y1 - prev_y) + (f11_x - f10_x) * (prev_y - y0);
            float dflowy_dx = (f01_y - f00_y) * (y1 - prev_y) + (f11_y - f10_y) * (prev_y - y0);
            float dflowx_dy = (f10_x - f00_x) * (x1 - prev_x) + (f11_x - f01_x) * (prev_x - x0);
            float dflowy_dy = (f10_y - f00_y) * (x1 - prev_x) + (f11_y - f01_y) * (prev_x - x0);

            // add grads wrt x and y point
            // TODO: this looks wrong, but is correct?
            float grad_point_x = grad_warped_point_x * (1 + dflowx_dx * -dz) + grad_warped_point_y * dflowy_dx * -dz;
            float grad_point_y = grad_warped_point_x * dflowx_dy * -dz + grad_warped_point_y * (1 + dflowy_dy * -dz);
            grad_warped_point_x = grad_point_x;
            grad_warped_point_y = grad_point_y;
        }
    }
}


torch::Tensor iterative_3d_warp_cuda(
    torch::Tensor points,
    torch::Tensor flow_fields,
    int num_warps, int threads, int points_per_thread) {

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

    iterative_3d_warp_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        flow_fields.data_ptr<float>(),
        warped_points.data_ptr<float>(),
        batch_size, num_points, num_flow_fields, num_z, num_warps, height, width);

    return warped_points;
}


std::vector<torch::Tensor> iterative_3d_warp_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor flow_fields,
    torch::Tensor warped_points,
    int num_warps, int threads, int points_per_thread) {

    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_flow_fields = flow_fields.size(1);
    int num_z = num_flow_fields + 1;
    int height = flow_fields.size(2);
    int width = flow_fields.size(3);

    auto grad_points = torch::zeros_like(points);
    auto grad_flow_fields = torch::zeros_like(flow_fields);

    // one or multiple points per thread
    int blocks = (batch_size * num_points + threads - 1) / threads / points_per_thread;

    iterative_3d_warp_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        points.data_ptr<float>(),
        flow_fields.data_ptr<float>(),
        warped_points.data_ptr<float>(),
        grad_points.data_ptr<float>(),
        grad_flow_fields.data_ptr<float>(),
        batch_size, num_points, num_flow_fields, num_z, num_warps, height, width);

    return {grad_points, grad_flow_fields};
}
