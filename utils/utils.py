import torch
import nibabel as nib
import nrrd
def numpy_reader(path):
    data = torch.from_numpy(nib.load(path).get_fdata()).float()
    affine = torch.eye(4, requires_grad=False)
    return data, affine

def nrrd_reader(path):
    d, h = nrrd.read(path)
    data = torch.from_numpy(d).float()
    affine = torch.eye(4, requires_grad=False)
    return data, affine