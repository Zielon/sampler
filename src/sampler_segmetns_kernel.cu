#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include "utils.h"

// Run per ray
__global__ void sample_segments_kernel(
    uint32_t n_elements,
    const float *depth,
    const float3 *points,
    const Tetra *flattend_tetras,
    const int *pix_to_face,
    const int *segments_per_ray,
    const int2 *tri_to_tetra,
    const int *packed_cumsum,
    int buffer_size,
    int n_rays,
    int n_tetras,
    float *t_starts,
    float *t_ends,
    int *tetra_indices,
    int *ray_indices,
    float4 *barys,
    int2 *packed_info)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements)
        return;

    packed_info[idx] = make_int2(packed_cumsum[idx], 0);

    int batch_id = int(idx / n_rays);
    int n_segments = segments_per_ray[idx];

    if (n_segments <= 1)
        return;

    const Tetra *tetras = flattend_tetras + batch_id * n_tetras;
    int segment_index = packed_cumsum[idx] - n_segments;
    packed_info[idx] = make_int2(segment_index, n_segments);

    for (int s = 0; s < n_segments - 1; ++s)
    {
        int tri_id = pix_to_face[idx * buffer_size + s];

        int2 tetra_neighbours = tri_to_tetra[tri_id];

        float3 point_start = points[idx * buffer_size + s];
        float3 point_end = points[idx * buffer_size + s + 1];

        float depth_start = depth[idx * buffer_size + s];
        float depth_end = depth[idx * buffer_size + s + 1];

        // float3 rd = normalize(point_end - point_start);
        // float3 ro = point_start;
        // float4 bary_coords;
        // int tetra_id = check_tetra_side(tetras, tetra_neighbours, ro, rd, bary_coords);

        // t_starts[segment_index + s] = depth_start;
        // t_ends[segment_index + s] = depth_end;
        // ray_indices[segment_index + s] = idx;    
        // tetra_indices[segment_index + s] = tetra_id;
        // barys[segment_index + s] = bary_coords;

        // printf("D: %f, S: %i, T: %i, 0: %i, 1: %i, 3D point [%f, %f, %f]\n",
        //     depth[idx * buffer_size + s], n_samples, tri_id, tetra_neighbours.x, tetra_neighbours.y, point_middle.x, point_middle.y, point_middle.z);
    }
}

// Run per batch
void run_sample_segments_kernel(
    const torch::Tensor &depth,
    const torch::Tensor &points,
    const torch::Tensor &pix_to_face,
    const torch::Tensor &tri_to_tetra,
    const torch::Tensor &tetra,
    const torch::Tensor &n_samples_per_ray,
    const torch::Tensor &packed_cumsum,
    const int n_samples,
    const int n_tetras,
    const int n_rays,
    const int buffer_size,
    float *t_starts,
    float *t_ends,
    int *tetra_indices,
    int *ray_indices,
    float *barys,
    int *packed_info,
    cudaStream_t stream)
{
    linear_kernel(sample_segments_kernel, 0, stream, n_rays,
                  // Input
                  depth.data_ptr<float>(),
                  reinterpret_cast<float3 *>(points.data_ptr<float>()),
                  reinterpret_cast<Tetra *>(tetra.data_ptr<float>()),
                  pix_to_face.data_ptr<int>(),
                  n_samples_per_ray.data_ptr<int>(),
                  reinterpret_cast<int2 *>(tri_to_tetra.data_ptr<int>()),
                  packed_cumsum.data_ptr<int>(),
                  buffer_size,
                  n_rays,
                  n_tetras,
                  // Output
                  t_starts,
                  t_ends,
                  tetra_indices,
                  ray_indices,
                  reinterpret_cast<float4 *>(barys),
                  reinterpret_cast<int2 *>(packed_info));
}
