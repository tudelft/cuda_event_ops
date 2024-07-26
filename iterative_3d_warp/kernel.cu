#include <torch/extension.h>
#include <vector>


__global__ void iterative_3d_warp_kernel(
    const float* __restrict__ points, 
    const float* __restrict__ flow_fields, 
    float* __restrict__ warped_points,
    int batch_size, int num_points, int num_flow_fields, int num_z, int height, int width,
    float x_min, float x_max, float y_min, float y_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_points) return;

    int batch_idx = idx / num_points;
    int point_idx = idx % num_points;

    // load point coordinates
    float x = points[batch_idx * num_points * 4 + point_idx * 4];
    float y = points[batch_idx * num_points * 4 + point_idx * 4 + 1];
    float z = points[batch_idx * num_points * 4 + point_idx * 4 + 2];

    // keep track of original z value, value and out of bounds status
    float val = points[batch_idx * num_points * 4 + point_idx * 4 + 3];
    float z_orig = z;
    bool is_out_of_bounds = false;

    // warp forward: increasing z values
    // start with next integer z value
    for (int z1 = static_cast<int>(ceil(z)); z1 < num_z; z1++) {
        int z0 = static_cast<int>(floor(z));
        float dz = z1 - z;

        // get the corresponding flow field
        const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z0 * height * width * 2;

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
        int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z * 5;
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
    x = points[batch_idx * num_points * 4 + point_idx * 4];
    y = points[batch_idx * num_points * 4 + point_idx * 4 + 1];
    z = points[batch_idx * num_points * 4 + point_idx * 4 + 2];

    // warp backward: decreasing z values
    // start with previous integer z value
    for (int z1 = static_cast<int>(floor(z)); z1 >= 0; z1--) {
        int z0 = static_cast<int>(ceil(z));
        float dz = z - z1;

        // get the corresponding flow field
        const float* flow_field = flow_fields + batch_idx * num_flow_fields * height * width * 2 + z0 * height * width * 2;

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
        int output_idx = batch_idx * num_points * num_z * 5 + point_idx * num_z * 5 + z * 5;
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
        batch_size, num_points, num_flow_fields, num_z, height, width,
        x_min, x_max, y_min, y_max);

    return warped_points;
}
