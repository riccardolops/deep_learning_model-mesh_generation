import torch
import nibabel as nib
import nrrd
import numpy as np
def nib_reader(path):
    data = torch.from_numpy(nib.load(path).get_fdata()).float()
    affine = data.affine
    return data, affine

def nrrd_reader(path):
    d, h = nrrd.read(path)
    data = torch.from_numpy(d.astype(np.int64)).float()
    rotation_matrix = h['space directions']
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix
    return data, affine
