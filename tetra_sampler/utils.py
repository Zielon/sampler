import numpy as np
import torch as th
import trimesh
import cv2

from pytorch3d.renderer import PerspectiveCameras


def save_tensor(tensor, name="test.png"):
    H, W, C = tensor.shape
    img = tensor.detach().float().cpu().numpy() * 255.0
    if C == 1:
        img = np.repeat(img, 3, axis=2)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_point_cloud(tensor, name="test.ply"):
    points = tensor.reshape(-1, 3).detach().cpu().numpy()
    colors = np.ones_like(points) * 55
    trimesh.PointCloud(vertices=points, colors=colors).export(name)


def repeat_cameras(cameras, times):
    R = th.repeat_interleave(cameras.R, times, dim=0)
    T = th.repeat_interleave(cameras.T, times, dim=0)
    fl = th.repeat_interleave(cameras.focal_length, times, dim=0)
    pp = th.repeat_interleave(cameras.principal_point, times, dim=0)
    image_size = th.repeat_interleave(cameras.image_size, times, dim=0)
    cameras = PerspectiveCameras(R=R, T=T, focal_length=fl, principal_point=pp, image_size=image_size, device=cameras.device)

    return cameras


def crop_image(image, crop_location):
    B = image.shape[0]
    cropped = []
    for i in range(B):
        crop_x_from, crop_x_to = crop_location[0][0][i].int(), crop_location[0][1][i].int()
        crop_y_from, crop_y_to = crop_location[1][0][i].int(), crop_location[1][1][i].int()

        cropped.append(image[i, crop_y_from:crop_y_to, crop_x_from:crop_x_to, :])

    return th.stack(cropped)
