#include <torch/extension.h>
#include <vector>


__global__ void iterative_3d_warp_kernel(
    const float* __restrict__ points, 
    const float* __restrict__ flow_fields, 
    float* __restrict__ warped_points,
    int num_points, int num_z, int height, int width,
    float x_min, float x_max, float y_min, float y_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // load point coordinates
    float x = points[idx * 4];
    float y = points[idx * 4 + 1];
    float z = points[idx * 4 + 2];

    // keep track of original z value, value and out of bounds status
    float val = points[idx * 4 + 3];
    float z_orig = z;
    bool is_out_of_bounds = false;

    // warp forward: increasing z values
    // start with next integer z value
    for (int z1 = static_cast<int>(ceil(z)); z1 < num_z; z1++) {
        int z0 = static_cast<int>(floor(z));
        float dz = z1 - z;

        // get the corresponding flow field
        const float* flow_field = flow_fields + z0 * height * width * 2;

        // bilinear interpolation to get flow at (x, y)
        int x0 = static_cast<int>(floor(x));
        int y0 = static_cast<int>(floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        x0 = max(0, min(x0, width - 1));
        y0 = max(0, min(y0, height - 1));
        x1 = max(0, min(x1, width - 1));
        y1 = max(0, min(y1, height - 1));

        float wa = (x1 - x) * (y1 - y);
        float wb = (x1 - x) * (y - y0);
        float wc = (x - x0) * (y1 - y);
        float wd = (x - x0) * (y - y0);

        float flow_x = wa * flow_field[(y0 * width + x0) * 2] +
                       wb * flow_field[(y1 * width + x0) * 2] +
                       wc * flow_field[(y0 * width + x1) * 2] +
                       wd * flow_field[(y1 * width + x1) * 2];

        float flow_y = wa * flow_field[(y0 * width + x0) * 2 + 1] +
                       wb * flow_field[(y1 * width + x0) * 2 + 1] +
                       wc * flow_field[(y0 * width + x1) * 2 + 1] +
                       wd * flow_field[(y1 * width + x1) * 2 + 1];
        
        // warp point position
        // scale flow by dz
        x += flow_x * dz;
        y += flow_y * dz;
        z = z1;  // need int; same as z += dz

        // save warped point position
        int output_idx = idx * num_z * 5 + z * 5;
        warped_points[output_idx] = x;
        warped_points[output_idx + 1] = y;
        warped_points[output_idx + 2] = z;
        warped_points[output_idx + 3] = z_orig;
        warped_points[output_idx + 4] = val;

        // check bounds
        if (x < x_min || x > x_max || y < y_min || y > y_max) {
            is_out_of_bounds = true;
            break;  // stop updating this point if it goes out of bounds
        }
    }

    // reload point coordinates
    x = points[idx * 4];
    y = points[idx * 4 + 1];
    z = points[idx * 4 + 2];

    // warp backward: decreasing z values
    // start with previous integer z value
    for (int z1 = static_cast<int>(floor(z)); z1 >= 0; z1--) {
        int z0 = static_cast<int>(ceil(z));
        float dz = z - z1;

        // get the corresponding flow field
        const float* flow_field = flow_fields + z0 * height * width * 2;

        // bilinear interpolation to get flow at (x, y)
        int x0 = static_cast<int>(floor(x));
        int y0 = static_cast<int>(floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        x0 = max(0, min(x0, width - 1));
        y0 = max(0, min(y0, height - 1));
        x1 = max(0, min(x1, width - 1));
        y1 = max(0, min(y1, height - 1));

        float wa = (x1 - x) * (y1 - y);
        float wb = (x1 - x) * (y - y0);
        float wc = (x - x0) * (y1 - y);
        float wd = (x - x0) * (y - y0);

        float flow_x = wa * flow_field[(y0 * width + x0) * 2] +
                       wb * flow_field[(y1 * width + x0) * 2] +
                       wc * flow_field[(y0 * width + x1) * 2] +
                       wd * flow_field[(y1 * width + x1) * 2];

        float flow_y = wa * flow_field[(y0 * width + x0) * 2 + 1] +
                       wb * flow_field[(y1 * width + x0) * 2 + 1] +
                       wc * flow_field[(y0 * width + x1) * 2 + 1] +
                       wd * flow_field[(y1 * width + x1) * 2 + 1];
        
        // warp point position
        // scale flow by dz
        x -= flow_x * dz;
        y -= flow_y * dz;
        z = z1;  // need int; same as z -= dz

        // save warped point position
        int output_idx = idx * num_z * 5 + z * 5;
        warped_points[output_idx] = x;
        warped_points[output_idx + 1] = y;
        warped_points[output_idx + 2] = z;
        warped_points[output_idx + 3] = z_orig;
        warped_points[output_idx + 4] = val;

        // check bounds
        if (x < x_min || x > x_max || y < y_min || y > y_max) {
            is_out_of_bounds = true;
            break;  // stop updating this point if it goes out of bounds
        }
    }

    // set all values to zero if out of bounds at some point
    if (is_out_of_bounds) {
        for (int z = 0; z < num_z; z++) {
            int output_idx = idx * num_z * 5 + z * 5;
            warped_points[output_idx + 4] = 0;
        }
    }
}


torch::Tensor iterative_3d_warp_cuda(
    torch::Tensor points,
    torch::Tensor flow_fields,
    float x_min, float x_max, float y_min, float y_max) {

    auto warped_points = torch::zeros({points.size(0), flow_fields.size(0) + 1, 5}, points.options());

    int threads = 1024;
    int blocks = (points.size(0) + threads - 1) / threads;

    iterative_3d_warp_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        flow_fields.data_ptr<float>(),
        warped_points.data_ptr<float>(),
        points.size(0), flow_fields.size(0) + 1, flow_fields.size(1), flow_fields.size(2),
        x_min, x_max, y_min, y_max);

    return warped_points;
}
