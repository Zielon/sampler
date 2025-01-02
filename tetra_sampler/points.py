import torch as th
from torch.autograd import Function

import sampling_points_cuda


class SamplePointsTetraCuda(Function):
    @staticmethod
    def forward(ctx, depth, pix_to_face, tetra_model, cage_vertices, segment_points, n_rays, n_max_samples, buffer_size, step_size):
        tri_to_tetra = tetra_model.triangle_to_tetra.int().contiguous()
        n_tetras = tetra_model.tetras.shape[0]
        tetras = cage_vertices[:, tetra_model.tetras].reshape(-1, 4, 3).contiguous()

        t_starts = th.zeros(n_max_samples).cuda().float()
        t_ends = th.zeros(n_max_samples).cuda().float()
        ray_indices = th.ones(n_max_samples).cuda().int()
        tetra_indices = th.zeros(n_max_samples).cuda().int()
        barys = th.zeros(n_max_samples, 4).cuda().float()
        sampled_points = th.zeros(n_max_samples, 3).cuda().float()
        packed_info = th.zeros(n_rays, 2).cuda().int()
        numsteps_counter = th.zeros(1).cuda().int()

        sampling_points_cuda.sample_forward(
            depth,
            segment_points,
            pix_to_face,
            tri_to_tetra,
            tetras,
            n_tetras,
            n_rays,
            n_max_samples,
            buffer_size,
            step_size,
            t_starts,
            t_ends,
            tetra_indices,
            ray_indices,
            barys,
            sampled_points,
            packed_info,
            numsteps_counter
        )

        n_samples = min(numsteps_counter.item(), n_max_samples)
        canonical_tetpoints = tetra_model.get_positions()[tetra_indices.long()]
        tetpoints = tetras[tetra_indices.long()]

        ctx.save_for_backward(
            tetpoints,
            canonical_tetpoints,
            sampled_points,
            barys,
            tetra_model,
            tetra_indices
        )

        return (
            sampled_points,
            t_starts,
            t_ends,
            tetra_indices.long(),
            ray_indices.long(),
            barys,
            packed_info.long()
        )

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        tetpoints, canonical_tetpoints, sampled_points, barys, tetra_model, tetra_indices = ctx.saved_tensors  
        grad_tetpoints = th.zeros_like(tetra_model.n(), 3)

        sampling_points_cuda.sample_backward(tetpoints, canonical_tetpoints, sampled_points, barys, grad_output.contiguous(), tetra_indices, grad_tetpoints)

        return None, None, None, grad_tetpoints, None, None, None, None, None, None, None, None

class PointsTetra():
    def __init__(self) -> None:
        pass

    @staticmethod
    def run(depth, pix_to_face, tetra_model, cage_vertices, segment_points, step_size=0.001, **kwargs):
        B, H, W, C = depth.shape
        n_max_samples = th.iinfo(th.int32).max
        if "n_max_samples" in kwargs:
            n_max_samples = kwargs["n_max_samples"]

        n_rays = H * W * B

        return SamplePointsTetraCuda.apply(
            depth.reshape(-1, C).contiguous(),
            pix_to_face.reshape(-1, C).int().contiguous(), 
            tetra_model,
            cage_vertices,
            segment_points.contiguous(),
            n_rays,
            n_max_samples,
            C,
            step_size
        )

def sample_occ_tet(tetra_model, cage_vertices, n_points_per_tet):
    n_tetras = tetra_model.tetras.shape[0]
    tetras = cage_vertices[:, tetra_model.tetras].reshape(-1, 4, 3).contiguous()

    sampled_points = th.zeros(n_points_per_tet * n_tetras, 3).cuda().float()

    sampling_points_cuda.sample_occ_tet(
        tetras,
        n_tetras,
        n_points_per_tet,
        sampled_points,
    )

    return sampled_points
