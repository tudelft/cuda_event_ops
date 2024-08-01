#include <torch/extension.h>
#include <vector>


__device__ bool check_out_of_bounds(
    float x, float y,
    float x_min, float x_max, float y_min, float y_max) {
    return (x < x_min || x >= x_max || y < y_min || y >= y_max);
}


__device__ void bilinear_interpolation(
    const float* flow_field,
    int height, int width, float x, float y,
    float& flow_x, float& flow_y) {

    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
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
    int batch_size, int num_points, int num_flow_fields, int num_z, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_points) return;

    int batch_idx = idx / num_points;
    int point_idx = idx % num_points;

    // load point coordinates
    float x = points[batch_idx * num_points * 4 + point_idx * 4];
    float y = points[batch_idx * num_points * 4 + point_idx * 4 + 1];
    float z = points[batch_idx * num_points * 4 + point_idx * 4 + 2];

    // keep track of original z value, value and out-of-bounds status
    float val = points[batch_idx * num_points * 4 + point_idx * 4 + 3];
    float z_orig = z;

    // if out of bounds to start with, return
    bool is_out_of_bounds = check_out_of_bounds(x, y, 0, width - 1, 0, height - 1);
    if (is_out_of_bounds) return;

    // warp forward: increasing z values
    // start with next integer z value
    for (int z1 = static_cast<int>(ceil(z)); z1 < num_z; z1++) {
        int z0 = static_cast<int>(floor(z));
        float dz = z1 - z;

        // bilinear interpolation to get flow at (x, y)
        const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z0 * height * width * 2;
        float flow_x, flow_y;
        bilinear_interpolation(flow_field, height, width, x, y, flow_x, flow_y);
        
        // warp point position
        // scale flow by dz
        x += flow_x * dz;
        y += flow_y * dz;
        z = z1;  // prevents rounding error?; same as z += dz

        // save warped point position
        int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z * 5;
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
        // reload point coordinates
        x = points[batch_idx * num_points * 4 + point_idx * 4];
        y = points[batch_idx * num_points * 4 + point_idx * 4 + 1];
        z = points[batch_idx * num_points * 4 + point_idx * 4 + 2];

        // warp backward: decreasing z values
        // start with previous integer z value
        for (int z1 = static_cast<int>(floor(z)); z1 >= 0; z1--) {
            int z0 = static_cast<int>(ceil(z));
            float dz = z - z1;

            // bilinear interpolation to get flow at (x, y)
            const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z0 * height * width * 2;
            float flow_x, flow_y;
            bilinear_interpolation(flow_field, height, width, x, y, flow_x, flow_y);

            // warp point position
            // scale flow by dz
            x -= flow_x * dz;
            y -= flow_y * dz;
            z = z1;  // need int; same as z -= dz

            // save warped point position
            int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z * 5;
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


__global__ void iterative_3d_warp_backward_kernel(
    const float* __restrict__ grad_output, 
    const float* __restrict__ points, 
    const float* __restrict__ flow_fields,
    const float* __restrict__ warped_points,
    float* __restrict__ grad_points,
    float* __restrict__ grad_flow_fields,
    int batch_size, int num_points, int num_flow_fields, int num_z, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_points) return;

    int batch_idx = idx / num_points;
    int point_idx = idx % num_points;

    // check if point was out of bounds: all values are zero
    if (warped_points[batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + 4] == 0) return;

    // load starting z
    // TODO: load from warped_points
    float z_orig = points[batch_idx * num_points * 4 + point_idx * 4 + 2];

    // iterate over z values in reverse for forward warping gradient computation
    for (int z1 = num_z - 1; z1 > z_orig; z1--) {
        int z0 = z1 - 1;
        float dz = min(1.0f, z1 - z_orig);

        // get warped point position
        int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z1 * 5;
        float x = warped_points[output_idx];
        float y = warped_points[output_idx + 1];

        // get bilinear interpolation weights
        int x0 = static_cast<int>(floor(x));
        int y0 = static_cast<int>(floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float w00 = (x1 - x) * (y1 - y);  // top left
        float w01 = (x - x0) * (y1 - y);  // top right
        float w10 = (x1 - x) * (y - y0);  // bottom left
        float w11 = (x - x0) * (y - y0);  // bottom right

        // get output gradients
        float grad_x = grad_output[output_idx];
        float grad_y = grad_output[output_idx + 1];

        // add output gradients to input gradients
        atomicAdd(&grad_points[batch_idx * num_points * 4 + point_idx * 4], grad_x);
        atomicAdd(&grad_points[batch_idx * num_points * 4 + point_idx * 4 + 1], grad_y);

        // add flow gradients in x
        
        // add flow gradients in y
    }

    // iterate over z values in reverse for backward warping gradient computation
    for (int z1 = 0; z1 < z_orig; z1++) {
        int z0 = z1 + 1;
        float dz = min(1.0f, z_orig - z1);

        // get warped point position
        int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z1 * 5;
        float x = warped_points[output_idx];
        float y = warped_points[output_idx + 1];

        // get bilinear interpolation weights
        int x0 = static_cast<int>(floor(x));
        int y0 = static_cast<int>(floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float w00 = (x1 - x) * (y1 - y);  // top left
        float w01 = (x - x0) * (y1 - y);  // top right
        float w10 = (x1 - x) * (y - y0);  // bottom left
        float w11 = (x - x0) * (y - y0);  // bottom right
        
        // get output gradients
        float grad_x = grad_output[output_idx];
        float grad_y = grad_output[output_idx + 1];

        // add output gradients to input gradients
        atomicAdd(&grad_points[batch_idx * num_points * 4 + point_idx * 4], grad_x);
        atomicAdd(&grad_points[batch_idx * num_points * 4 + point_idx * 4 + 1], grad_y);

        // add flow gradients in x

        // add flow gradients in y
    }
}


torch::Tensor iterative_3d_warp_cuda(
    torch::Tensor points,
    torch::Tensor flow_fields) {

    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_flow_fields = flow_fields.size(1);
    int num_z = num_flow_fields + 1;
    int height = flow_fields.size(2);
    int width = flow_fields.size(3);

    // points: (b, n, 4)
    // flow_fields: (b, d, h, w, 2)
    // warped_points: (b, n, d + 1, 5)
    auto warped_points = torch::zeros({batch_size, num_points, num_z, 5}, points.options());

    int threads = 1024;
    int blocks = (batch_size * num_points + threads - 1) / threads;

    iterative_3d_warp_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        flow_fields.data_ptr<float>(),
        warped_points.data_ptr<float>(),
        batch_size, num_points, num_flow_fields, num_z, height, width);

    return warped_points;
}


std::vector<torch::Tensor> iterative_3d_warp_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor flow_fields,
    torch::Tensor warped_points) {

    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_flow_fields = flow_fields.size(1);
    int num_z = num_flow_fields + 1;
    int height = flow_fields.size(2);
    int width = flow_fields.size(3);

    auto grad_points = torch::zeros_like(points);
    auto grad_flow_fields = torch::zeros_like(flow_fields);

    int threads = 1024;
    int blocks = (batch_size * num_points + threads - 1) / threads;

    iterative_3d_warp_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        points.data_ptr<float>(),
        flow_fields.data_ptr<float>(),
        warped_points.data_ptr<float>(),
        grad_points.data_ptr<float>(),
        grad_flow_fields.data_ptr<float>(),
        batch_size, num_points, num_flow_fields, num_z, height, width);

    return {grad_points, grad_flow_fields};
}
