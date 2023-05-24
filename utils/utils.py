import torch
from scipy.ndimage import distance_transform_edt
import numpy as np
import torchio as tio

def bin2dist(mask):
    dist_transform = distance_transform_edt(mask.data[0].bool(), sampling=[abs(mask.affine[0][0]), abs(mask.affine[1][1]), abs(mask.affine[2][2])])
    dist_transform_inv = -1 * distance_transform_edt(~mask.data[0].bool(), sampling=[abs(mask.affine[0][0]), abs(mask.affine[1][1]), abs(mask.affine[2][2])])
    dist_transform = dist_transform + dist_transform_inv
    ma = np.maximum(abs(dist_transform.min()), abs(dist_transform.max()))
    rescale = tio.transforms.RescaleIntensity(out_min_max=(-1, 1), in_min_max=(-ma, ma))
    dist_transform = torch.tensor(dist_transform).unsqueeze(0)
    dist_transform = rescale(dist_transform)
    dist_transform = dist_transform.squeeze().numpy()
    dist = tio.ScalarImage(path=mask.path, tensor=np.expand_dims(dist_transform, axis=0), affine=mask.affine)
    return dist
