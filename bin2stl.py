import os

import mcubes
import open3d as o3d
import pyvista as pv
import numpy as np
import pyacvd
import trimesh
from trimesh.exchange.off import export_off
import torchio as tio
from skeletonize import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path_origin_mask = '/home/rlops/datasets/AVT/Dongyang/D2/D2.seg.nrrd'
image = tio.LabelMap(path_origin_mask)
transform = tio.Resample(1)
transformed = transform(image)

mcs_vert, mcs_tri = mcubes.marching_cubes(transformed.data.numpy()[0], 0)
mcs_mesh = o3d.geometry.TriangleMesh()
mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

otmesh = trimesh.Trimesh(np.asarray(mcs_mesh.vertices), faces=np.asarray(mcs_mesh.triangles)).apply_scale(1)
output_file = 'D2.off'
off_data = export_off(otmesh, 0)
with open(output_file, 'w') as file:
    file.write(off_data)
pmesh = pv.wrap(otmesh)
pmesh.save("mesh" + ".stl")

skel = skeletonize(speed_power=1.2, Euler_step_size=0.5, depth_th=4, length_th=None, segments_th=3, simple_path=False, verbose=True)
skeletons = skel.skeleton(transformed.data.numpy()[0])
d_mat = []
for sk in skeletons:
    d_mat.append(sk)
ske_pointcld = np.concatenate(d_mat)
pv.global_theme.color_cycler = 'default'
p = pv.Plotter()
for sk in skeletons:
    p.add_points(sk)
p.add_mesh(pmesh, opacity=0.10)
p.show()
np.save(os.path.splitext(os.path.splitext(path_origin_mask)[0])[0] + '_skl.npy', ske_pointcld)
pmesh.save(os.path.splitext(os.path.splitext(path_origin_mask)[0])[0] + '_model.ply')





