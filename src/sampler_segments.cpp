#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <vector>

#include "common.h"

void run_sample_segments_kernel(    
    const torch::Tensor &depth,
    const torch::Tensor &points,
    const torch::Tensor &pix_to_face, 
    const torch::Tensor &tri_to_tetra, 
    const torch::Tensor &tris,
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
    cudaStream_t stream
);

std::vector<torch::Tensor> sample_segments(        
    torch::Tensor depth, 
    torch::Tensor points,
    torch::Tensor pix_to_face, 
    torch::Tensor tri_to_tetra, 
    torch::Tensor tetra,
    torch::Tensor n_samples_per_ray,
    torch::Tensor packed_cumsum,
    int n_samples,
    int n_tetras,
    int n_rays,
    int buffer_size,
    torch::Tensor t_starts,
    torch::Tensor t_ends,
    torch::Tensor tetra_indices,
    torch::Tensor ray_indices,
    torch::Tensor barys,
    torch::Tensor packed_info)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    CHECK_INPUT(depth);
    CHECK_INPUT(points);
    CHECK_INPUT(pix_to_face);
    CHECK_INPUT(tri_to_tetra);
    CHECK_INPUT(tetra);
    CHECK_INPUT(n_samples_per_ray);
    CHECK_INPUT(packed_cumsum);
    CHECK_INPUT(t_starts);
    CHECK_INPUT(t_ends);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(tetra_indices);
    CHECK_INPUT(barys);
    CHECK_INPUT(packed_info);

    run_sample_segments_kernel(
        depth,
        points,
        pix_to_face,
        tri_to_tetra,
        tetra,
        n_samples_per_ray,
        packed_cumsum,
        n_samples,
        n_tetras,
        n_rays,
        buffer_size,
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>(),
        tetra_indices.data_ptr<int>(),
        ray_indices.data_ptr<int>(),
        barys.data_ptr<float>(),
        packed_info.data_ptr<int>(),
        stream
    );

    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sample", &sample_segments, "Sample segmetns using rasterizer");
}
