import torch as th
from torch.autograd import Function

from einops import rearrange

import sampling_segments_cuda


class SampleSegmentsTetraCuda(Function):
    @staticmethod
    def forward(ctx, depth, pix_to_face, tri_to_tetra, tetras, n_segments_per_ray, packed_cumsum, points, n_samples, n_tetras, n_rays, buffer_size):
        t_starts = th.zeros(n_samples).cuda().float()
        t_ends = th.zeros(n_samples).cuda().float()
        ray_indices = th.ones(n_samples).cuda().int()
        tetra_indices = th.zeros(n_samples).cuda().int()
        barys = th.zeros(n_samples, 4).cuda().float()
        packed_info = th.zeros(n_rays, 2).cuda().int()

        sampling_segments_cuda.sample(
            depth,
            points,
            pix_to_face,
            tri_to_tetra,
            tetras,
            n_segments_per_ray,
            packed_cumsum,
            n_samples,
            n_tetras,
            n_rays,
            buffer_size,
            t_starts,
            t_ends,
            tetra_indices,
            ray_indices,
            barys,
            packed_info
        )

        return t_starts, t_ends, tetra_indices.long(), ray_indices.long(), barys, packed_info.long()

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class SegmentsTetra():
    def __init__(self) -> None:
        pass

    @staticmethod
    def run(depth, pix_to_face, tri_to_tetra, tetras, n_tetras, points, **kwargs):
        B, H, W, C = depth.shape
        valid_samples = pix_to_face >= 0

        n_segments_per_ray = valid_samples.sum(dim=3, keepdim=True)
        n_segments_per_ray = n_segments_per_ray.reshape(-1).int().contiguous()
        n_segments_per_ray[n_segments_per_ray <= 1] = 0
        n_samples = n_segments_per_ray.sum().item()
        packed_cumsum = th.cumsum(n_segments_per_ray, dim=0)
        n_rays = H * W * B

        return SampleSegmentsTetraCuda.apply(
            depth.reshape(-1, C).contiguous(),
            pix_to_face.reshape(-1, C).int().contiguous(), 
            tri_to_tetra.int().contiguous(), 
            tetras.reshape(-1, 4, 3).contiguous(),
            n_segments_per_ray,
            packed_cumsum.int().contiguous(),
            points.contiguous(),
            n_samples,
            n_tetras,
            n_rays,
            C
        )
