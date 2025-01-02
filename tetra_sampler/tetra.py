from collections import OrderedDict
import os
from tqdm import tqdm

import meshio
import trimesh
import numpy as np
import torch
import torch.nn as nn


class Tetra:
    def __init__(self, path) -> None:
        super(Tetra, self).__init__()
        mesh = meshio.read(path)
        # meshio.write(path.replace("mesh", "vtk"), mesh)

        self.tetras = torch.from_numpy(mesh.cells_dict['tetra']).cuda()

        tris = mesh.cells_dict['triangle']
        mesh = trimesh.Trimesh(vertices=mesh.points, faces=tris, process=False)
        trimesh.repair.fix_normals(mesh)

        self.triangles = torch.from_numpy(mesh.faces).cuda()
        self.points = torch.from_numpy(mesh.vertices).float().cuda()

        self.A = self.tetras[:, 0]
        self.B = self.tetras[:, 1]
        self.C = self.tetras[:, 2]
        self.D = self.tetras[:, 3]

        v0 = torch.stack([self.A, self.B, self.C], dim=1)
        v1 = torch.stack([self.A, self.B, self.D], dim=1)
        v2 = torch.stack([self.A, self.C, self.D], dim=1)
        v3 = torch.stack([self.B, self.C, self.D], dim=1)

        self.trimesh = mesh
        self.ABCD = torch.stack([v0, v1, v2, v3], dim=1)

        topology_path = path.replace("cage.mesh", "topology.npy")
        faces_path = path.replace("cage.mesh", "faces.npy")
        if os.path.exists(topology_path):
            self.tetra_faces = torch.from_numpy(np.load(faces_path)).cuda()
            self.triangle_to_tetra = torch.from_numpy(np.load(topology_path)).int().cuda()
        else:
            # Add only unique faces
            all_faces = torch.sort(self.ABCD.view(-1, 3), dim=1)[0]
            self.tetra_faces = torch.unique(all_faces, dim=0)
            topology = self.build()

            np.save(topology_path, topology)
            np.save(faces_path, self.tetra_faces.cpu().numpy())
            self.triangle_to_tetra = torch.from_numpy(topology).int().cuda()
            
            torch.cuda.empty_cache()

        self.tatra_color = np.array([np.random.choice(range(255), size=3) / 255 for _ in range(self.tetras.shape[0])])
        self.tetra_trimesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=self.tetra_faces.cpu().numpy(), process=False
        )
        self.surface_triangle = self.tetra_faces[self.triangle_to_tetra[:, 1] == -1]
        self.surface_tetras = self.triangle_to_tetra[self.triangle_to_tetra[:, 1] == -1][:, 0].long()

    def get_positions(self, points=None):
        if points is None:
            points = self.points
        return points[self.tetras]

    def get_triangles(self, points=None):
        if points is None:
            points = self.points
        return points[self.tetra_faces]

    def gradient(self, tetras):
        v0 = tetras[:, 0]
        v1 = tetras[:, 1]
        v2 = tetras[:, 2]
        v3 = tetras[:, 3]

        return torch.stack([v3 - v0, v2 - v0, v1 - v0], dim=2)

    def n(self):
        return self.tetras.shape[0]

    def n_points(self):
        return self.points.size(0)

    def build(self):
        tetra_faces = self.tetra_faces.cpu().numpy()
        N = self.tetra_faces.shape[0]

        face_to_ids = {}
        for face_id in range(N):
            key = hash(tetra_faces[face_id].tostring())
            face_to_ids[key] = face_id

        sorted_tetras = torch.sort(self.tetras, dim=1)[0].cpu().numpy()
        N = sorted_tetras.shape[0]
        face_to_tetra = {}

        def add_face(face, tet_id):
            key = hash(face.tostring())
            if key in face_to_tetra:
                face_to_tetra[key].append(tet_id)
            else:
                face_to_tetra[key] = [tet_id]

        for tet_id in tqdm(range(N)):
            tetra = sorted_tetras[tet_id]
            a, b, c, d = tetra[0], tetra[1], tetra[2], tetra[3]
            f0 = np.sort(np.array([a, b, c]))
            f1 = np.sort(np.array([a, b, d]))
            f2 = np.sort(np.array([a, c, d]))
            f3 = np.sort(np.array([b, c, d]))

            add_face(f0, tet_id)
            add_face(f1, tet_id)
            add_face(f2, tet_id)
            add_face(f3, tet_id)

        triangle_to_tetra = np.ones((self.tetra_faces.shape[0], 2)).astype(np.int32) * -1

        for key in face_to_tetra.keys():
            i = 0
            for tet_id in face_to_tetra[key]:
                face_id = face_to_ids[key]
                triangle_to_tetra[face_id, i] = tet_id
                i += 1

        return triangle_to_tetra
