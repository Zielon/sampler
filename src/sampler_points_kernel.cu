#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include "utils.h"

pcg32 m_rng{};

// Run per ray
__global__ void sample_points_kernel_forward(
    // Input
    uint32_t n_elements,
    const float *depth,
    const float3 *points,
    const Tetra *flattend_tetras,
    const int *pix_to_face,
    const int2 *tri_to_tetra,
    int buffer_size,
    float step_size,
    int n_rays,
    int n_max_samples,
    int n_tetras,
    int *numsteps_counter,
    // Output
    float *t_starts,
    float *t_ends,
    int *tetra_indices,
    int *ray_indices,
    float4 *barys,
    float3 *sampled_points,
    int2 *packed_info)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements)
        return;

    int batch_id = int(idx / n_rays);

    const Tetra *tetras = flattend_tetras + batch_id * n_tetras;

    // first pass to compute an accurate number of steps
    int j = 0;
    for (int s = 1; s < buffer_size; ++s)
    {
        int current = idx * buffer_size + s;

        int tri_start = pix_to_face[current - 1];
        int tri_end = pix_to_face[current];
        if (tri_start == -1 || tri_end == -1)
            break; // Nothing more to traverse

        float3 segment_start = points[current - 1];
        float3 segment_end = points[current];

        float depth_start = depth[current - 1];
        float depth_end = depth[current];

        float segment_length = depth_end - depth_start;
        float3 rd = normalize(segment_end - segment_start);
        float3 ro = segment_start;

        int tetra_id = get_tetra_id(tri_to_tetra[tri_start], tri_to_tetra[tri_end], tetras, ro + rd * segment_length / 2.f);
        if (tetra_id == -1)
            continue; // End of the segment

        float t = EPS;
        while (t + step_size <= segment_length)
        {
            t += step_size;
            ++j;
        }
    }

    int numsteps = j;
    int base = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter
    if (base + numsteps > n_max_samples)
    {
        return;
    }

    // second pass write out all the values

    packed_info[idx] = make_int2(base, numsteps);

    j = 0;
    for (int s = 1; s < buffer_size; ++s)
    {
        int current = idx * buffer_size + s;

        int tri_start = pix_to_face[current - 1];
        int tri_end = pix_to_face[current];
        if (tri_start == -1 || tri_end == -1)
            break; // Nothing more to traverse

        float3 segment_start = points[current - 1];
        float3 segment_end = points[current];

        float depth_start = depth[current - 1];
        float depth_end = depth[current];

        float segment_length = depth_end - depth_start;
        float3 rd = normalize(segment_end - segment_start);
        float3 ro = segment_start;

        int tetra_id = get_tetra_id(tri_to_tetra[tri_start], tri_to_tetra[tri_end], tetras, ro + rd * segment_length / 2.f);
        if (tetra_id == -1)
            continue; // End of the segment

        // Current tetra
        Tetra tetra = tetras[tetra_id];
        float t = EPS;
        while (t + step_size <= segment_length)
        {
            float3 pos = ro + rd * t;
            int i = base + j;

            t_starts[i] = depth_start + t;
            t_ends[i] = depth_start + t + step_size;
            ray_indices[i] = idx;
            tetra_indices[i] = tetra_id;
            sampled_points[i] = pos;
            barys[i] = barycentric(pos, tetra);

            t += step_size;
            ++j;
        }
    }
}

// Run per batch
void run_sample_points_kernel_forward(
    const torch::Tensor &depth,
    const torch::Tensor &points,
    const torch::Tensor &pix_to_face,
    const torch::Tensor &tri_to_tetra,
    const torch::Tensor &tetras,
    const int n_tetras,
    const int n_rays,
    const int n_max_samples,
    const int buffer_size,
    const float step_size,
    float *t_starts,
    float *t_ends,
    int *tetra_indices,
    int *ray_indices,
    float *barys,
    float *sampled_points,
    int *packed_info,
    int *numsteps_counter,
    cudaStream_t stream)
{
    linear_kernel(sample_points_kernel_forward, 0, stream, n_rays,
                  // Input
                  depth.data_ptr<float>(),
                  reinterpret_cast<float3 *>(points.data_ptr<float>()),
                  reinterpret_cast<Tetra *>(tetras.data_ptr<float>()),
                  pix_to_face.data_ptr<int>(),
                  reinterpret_cast<int2 *>(tri_to_tetra.data_ptr<int>()),
                  buffer_size,
                  step_size,
                  n_rays,
                  n_max_samples,
                  n_tetras,
                  numsteps_counter,
                  // Output
                  t_starts,
                  t_ends,
                  tetra_indices,
                  ray_indices,
                  reinterpret_cast<float4 *>(barys),
                  reinterpret_cast<float3 *>(sampled_points),
                  reinterpret_cast<int2 *>(packed_info));
}

__global__ void sample_occ_tet_kernel_forward(
    uint32_t n_elements,
    const Tetra *tetras,
    const int n_tetras,
    const int n_points_per_tet,
    float3 *sampled_points,
    pcg32 rng)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements)
        return;

    Tetra tetra = tetras[idx];

    for (int i = 0; i < n_points_per_tet; ++i)
    {
        rng.advance(idx * 3);

        float s = rng.next_float();
        float t = rng.next_float();
        float u = rng.next_float();

        sampled_points[idx + i] = pick_tet(tetra, s, t, u);
    }
}

void run_sample_occ_tet_kernel_forward(
    const torch::Tensor &tetras,
    const int n_tetras,
    const int n_points_per_tet,
    float *sampled_points,
    cudaStream_t stream)
{
    linear_kernel(sample_occ_tet_kernel_forward, 0, stream, n_tetras,
        reinterpret_cast<Tetra *>(tetras.data_ptr<float>()),
        n_tetras,
        n_points_per_tet,
        reinterpret_cast<float3 *>(sampled_points),
        m_rng
    );

    m_rng.advance();
}

__global__ void sample_points_kernel_backward(
    uint32_t n_elements,
    const Tetra *tetpoints,
    const Tetra *canonical_tetpoints,
    const float3 *sampled_points,
    const float4 *barys,
    const float *grad_output,
    const int *tetra_indices,
    float3 *grad_tetpoints)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements)
        return;

    Tetra tetra_deformed = tetpoints[idx];
    Tetra tetra_canonical = canonical_tetpoints[idx];
    float3 p0 = sampled_points[idx];
    float4 bary = barys[idx];
    int tetra_id = tetra_indices[idx];

    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 0)*3 + 0, grad_v0.x);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 0)*3 + 1, grad_v0.y);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 0)*3 + 2, grad_v0.z);

    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 1)*3 + 0, grad_v1.x);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 1)*3 + 1, grad_v1.y);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 1)*3 + 2, grad_v1.z);

    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 2)*3 + 0, grad_v2.x);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 2)*3 + 1, grad_v2.y);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 2)*3 + 2, grad_v2.z);

    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 3)*3 + 0, grad_v3.x);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 3)*3 + 1, grad_v3.y);
    // atomicAdd((float*)grad_tetpoints + ((n * K + k) * 4 + 3)*3 + 2, grad_v3.z);
}

void run_sample_points_kernel_backward(
    const torch::Tensor &tetpoints,
    const torch::Tensor &canonical_tetpoints,
    const torch::Tensor &sampled_points,
    const torch::Tensor &bary,
    const torch::Tensor &grad_output,
    const torch::Tensor &tetra_indices,
    float *grad_tetpoints,
    cudaStream_t stream)
{
    int n_points = sampled_points.size(0);
    linear_kernel(sample_points_kernel_backward, 0, stream, n_points,
                  reinterpret_cast<Tetra *>(tetpoints.data_ptr<float>()),
                  reinterpret_cast<Tetra *>(canonical_tetpoints.data_ptr<float>()),
                  reinterpret_cast<float3 *>(sampled_points.data_ptr<float>()),
                  reinterpret_cast<float4 *>(bary.data_ptr<float>()),
                  reinterpret_cast<float *>(grad_output.data_ptr<float>()),
                  reinterpret_cast<int *>(tetra_indices.data_ptr<int>()),
                  reinterpret_cast<float3 *>(grad_tetpoints));
}

__global__ void barycentric_kernel(
    uint32_t n_elements,
    const float3 *points,
    const Tetra *tetras,
    const int *closest_faces,
    const int2 *tri_to_tetra,
    float4 *barys,
    int *tetra_id,
    int *active_points)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements)
        return;

    int face_id = closest_faces[idx];
    float3 point = points[idx];
    int id = get_id(tri_to_tetra[face_id], tetras, point);

    // printf("face id = %i, tet id = %i, p = [%f, %f, %f]\n", face_id, id, point.x, point.y, point.z)

    if (id == -1)
    {
        barys[idx] = make_float4(0);
        active_points[idx] = -1;
        tetra_id[idx] = -1;
        return;
    }

    Tetra tetra = tetras[id];
    float4 bary = barycentric(point, tetra);

    barys[idx] = bary;
    tetra_id[idx] = id;
    active_points[idx] = 1;
}

void run_barycentric_kernel(
    const torch::Tensor &tetras,
    const torch::Tensor &points,
    const torch::Tensor &closest_faces,
    const torch::Tensor &tri_to_tetra,
    float *barys,
    int *tetra_id,
    int *active_points,
    cudaStream_t stream)
{
    int n_points = points.size(0);

    linear_kernel(barycentric_kernel, 0, stream, n_points,
                // Input
                reinterpret_cast<float3 *>(points.data_ptr<float>()),
                reinterpret_cast<Tetra *>(tetras.data_ptr<float>()),
                reinterpret_cast<int *>(closest_faces.data_ptr<int>()),
                reinterpret_cast<int2 *>(tri_to_tetra.data_ptr<int>()),
                reinterpret_cast<float4 *>(barys),
                tetra_id,
                active_points);
}
