import pytorch3d
try:
    import kornia
    kornia_available = True
except ImportError:
    kornia_available = False
import math
import numpy as np
import torch as th
import torch.nn as nn
from scipy.spatial.transform import Rotation
from pytorch3d.utils import cameras_from_opencv_projection, opencv_from_cameras_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras

# from pytorch3d.renderer.opengl.rasterizer_opengl import (
#     MeshRasterizerOpenGL
# )


class Renderer(nn.Module):
    def __init__(self, raster_settings: RasterizationSettings):
        super().__init__()
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)

    @staticmethod
    def sample_from_mask(alpha, height, width, random_crop):
        B, H, W, C = alpha.shape
        Y, X = random_crop
        kernel = th.ones(50, 50).cuda()
        if kornia_available:
            alpha = kornia.morphology.dilation(alpha.permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
        mask = alpha.sum(dim=0) > 0
        mask = mask.reshape(-1, 1).squeeze()
        x, y = th.meshgrid(
            th.arange(width, device=alpha.device),
            th.arange(height, device=alpha.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()
        active = mask.sum().item()
        idx = th.randperm(active)[:B]
        xy_b = th.stack([x, y], dim=1)[mask][idx]

        x = th.clip(xy_b[:, 0], 0, width - X)
        y = th.clip(xy_b[:, 1], 0, height - Y)

        return x, y

    @staticmethod
    def crop_camera(random_crop, K, image_size, sample_mask):
        K = K.clone()
        B = image_size.shape[0]
        height, width = image_size[0, 0].item(), image_size[0, 1].item()
        Y, X = random_crop
        cropped_image_size = th.tensor([[Y, X]]).expand(B, -1).cuda()

        if sample_mask is not None:
            x, y = Renderer.sample_from_mask(sample_mask, height, width, random_crop)
        else:
            x = th.randint(0, width - X + 1, (B,)).cuda()
            y = th.randint(0, height - Y + 1, (B,)).cuda()

        K[:, 0, 2] -= x
        K[:, 1, 2] -= y

        crop_location = (x, x + X), (y, y + Y)

        return K, cropped_image_size, crop_location

    @staticmethod
    def camera_to_patch(Rt, K, image_size, patch_size=128):
        H, W = image_size
        R = Rt[:, :3, :3]
        tvec = Rt[:, :3, 3]

        rows = []

        hs = np.arange(0, H, patch_size)
        ws = np.arange(0, W, patch_size)

        for i, y in enumerate(hs):  # rows
            columns = []
            for j, x in enumerate(ws):  # columns
                k = K.clone()

                target_width = W - x if j == len(ws) - 1 else patch_size
                target_height = H - y if i == len(hs) - 1 else patch_size

                k[:, 0, 2] -= x
                k[:, 1, 2] -= y

                crop = (th.Tensor([x]), th.Tensor([x + target_width])), (th.Tensor([y]), th.Tensor([y + target_height]))

                size = th.Tensor([[target_height, target_width]]).cuda().int()
                C = cameras_from_opencv_projection(R, tvec, k, size)
                # r, t, k = opencv_from_cameras_projection(C, size)

                columns.append((C, crop))

            rows.append(columns)

        return rows

    @staticmethod
    def to_cameras(Rt, K, image_size, random_crop=None, sample_mask=None) -> PerspectiveCameras:
        R = Rt[:, :3, :3]
        tvec = Rt[:, :3, 3]
        crop_location = None
        if random_crop is not None:
            K, image_size, crop_location = Renderer.crop_camera(
                random_crop, K, image_size, sample_mask
            )
        cameras = cameras_from_opencv_projection(R, tvec, K, image_size)

        return cameras, crop_location

    def forward(self, cameras, vertices, faces):
        meshes = Meshes(verts=vertices.float(), faces=faces.long())

        fragmetns = self.rasterizer(meshes, cameras=cameras)

        return fragmetns
