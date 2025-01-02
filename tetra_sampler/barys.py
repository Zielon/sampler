import torch as th
from typing import Dict, Optional, Tuple
from torch.autograd import Function

import sampling_points_cuda
import bvh_distance_queries


class BarycentricCuda(Function):
    @staticmethod
    def forward(ctx, points, tetras, triangles, tri_to_tetra, tetra_model):
        n = points.size(0)

        bvh = bvh_distance_queries.BVH()
        closest_faces = bvh(triangles[None], points[None])[2][0].int()

        barys = th.zeros(n, 4).float().cuda()
        tetra_id = th.zeros(n).int().cuda()
        active_points = th.zeros(n).int().cuda()

        sampling_points_cuda.compute_bary(tetras, points, closest_faces, tri_to_tetra, barys, tetra_id, active_points)

        tetra_id = tetra_id.long()

        return barys, tetra_id, active_points

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


def compute_bary(points, tetras, triangles, tri_to_tetra, tetra_model):
    return BarycentricCuda.apply(points, tetras, triangles, tri_to_tetra, tetra_model)
