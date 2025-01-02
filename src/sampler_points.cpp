#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <vector>

#include "common.h"

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
    cudaStream_t stream);

std::vector<torch::Tensor> sample_points_forward(
    torch::Tensor depth,
    torch::Tensor points,
    torch::Tensor pix_to_face,
    torch::Tensor tri_to_tetra,
    torch::Tensor tetras,
    int n_tetras,
    int n_rays,
    int n_max_samples,
    int buffer_size,
    float step_size,
    torch::Tensor t_starts,
    torch::Tensor t_ends,
    torch::Tensor tetra_indices,
    torch::Tensor ray_indices,
    torch::Tensor barys,
    torch::Tensor sampled_points,
    torch::Tensor packed_info,
    torch::Tensor numsteps_counter)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    CHECK_INPUT(depth);
    CHECK_INPUT(points);
    CHECK_INPUT(pix_to_face);
    CHECK_INPUT(tri_to_tetra);
    CHECK_INPUT(tetras);
    CHECK_INPUT(t_starts);
    CHECK_INPUT(t_ends);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(tetra_indices);
    CHECK_INPUT(barys);
    CHECK_INPUT(sampled_points);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(numsteps_counter);

    run_sample_points_kernel_forward(
        depth,
        points,
        pix_to_face,
        tri_to_tetra,
        tetras,
        n_tetras,
        n_rays,
        n_max_samples,
        buffer_size,
        step_size,
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>(),
        tetra_indices.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        barys.data_ptr<float>(),
        sampled_points.data_ptr<float>(),
        packed_info.data_ptr<int>(),
        numsteps_counter.data_ptr<int>(),
        stream);

    return {};
}

void run_sample_occ_tet_kernel_forward(
    const torch::Tensor &tetras,
    const int n_tetras,
    const int n_points_per_tet,
    float *sampled_points,
    cudaStream_t stream);

std::vector<torch::Tensor> sample_occ_tet_kernel_forward(
    torch::Tensor tetras,
    int n_tetras,
    int n_points_per_tet,
    torch::Tensor sampled_points)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    CHECK_INPUT(tetras);
    CHECK_INPUT(sampled_points);

    run_sample_occ_tet_kernel_forward(
        tetras,
        n_tetras,
        n_points_per_tet,
        sampled_points.data_ptr<float>(),
        stream);

    return {};
}

void run_sample_points_kernel_backward(
    const torch::Tensor &tetpoints,
    const torch::Tensor &canonical_tetpoints,
    const torch::Tensor &sampled_points,
    const torch::Tensor &bary,
    const torch::Tensor &grad_output,
    const torch::Tensor &tetra_indices,
    float *grad_tetpoints,
    cudaStream_t stream);

std::vector<torch::Tensor> sample_points_backward(
    torch::Tensor tetpoints,
    torch::Tensor canonical_tetpoints,
    torch::Tensor sampled_points,
    torch::Tensor bary,
    torch::Tensor grad_output,
    torch::Tensor tetra_indices,
    torch::Tensor grad_tetpoints)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // auto stream = std::make_unique<StreamAndEvent>()->get();

    CHECK_INPUT(tetpoints);
    CHECK_INPUT(canonical_tetpoints);
    CHECK_INPUT(sampled_points);
    CHECK_INPUT(bary);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(tetra_indices);
    CHECK_INPUT(grad_tetpoints);

    run_sample_points_kernel_backward(
        tetpoints,
        canonical_tetpoints,
        sampled_points,
        bary,
        grad_output,
        tetra_indices,
        grad_tetpoints.data_ptr<float>(),
        stream);

    return {};
}

void run_barycentric_kernel(
    const torch::Tensor &tetras,
    const torch::Tensor &points,
    const torch::Tensor &closest_faces,
    const torch::Tensor &tri_to_tetra,
    float *barys,
    int *tetra_id,
    int *active_points,
    cudaStream_t stream
);

std::vector<torch::Tensor> compute_bary(
    torch::Tensor tetras,
    torch::Tensor points,
    torch::Tensor closest_faces,
    torch::Tensor tri_to_tetra,
    torch::Tensor barys,
    torch::Tensor tetra_id,
    torch::Tensor active_points)
{
    CHECK_INPUT(tetras);
    CHECK_INPUT(points);
    CHECK_INPUT(closest_faces);
    CHECK_INPUT(tri_to_tetra);
    CHECK_INPUT(barys);
    CHECK_INPUT(tetra_id);
    CHECK_INPUT(active_points);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    run_barycentric_kernel(
        tetras,
        points,
        closest_faces,
        tri_to_tetra,
        barys.data_ptr<float>(),
        tetra_id.data_ptr<int>(),
        active_points.data_ptr<int>(),
        stream
    );

    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sample_forward", &sample_points_forward, "Sample points using rasterizer forward");
    m.def("sample_backward", &sample_points_backward, "Sample points using rasterizer backward");
    m.def("sample_occ_tet", &sample_occ_tet_kernel_forward, "Sample points from tet mesh");
    m.def("compute_bary", &compute_bary, "Compute bary inside tetras");
}
