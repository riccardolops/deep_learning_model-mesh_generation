import numpy as np
import torch
from dataloader.AugFactory import AugFactory
import torchio as tio
import yaml
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import distance_transform_edt
from utils.utils import bin2dist_disp
import torch.nn.functional as F
from skimage import segmentation as skimage_seg
from skeletonize import skeletonize

matplotlib.use('GTK3Agg')

with open('configs/preprocessing.yaml', 'r') as preproc_file:
    preproc = yaml.load(preproc_file, yaml.FullLoader)
preprocessing = AugFactory(preproc).get_transform()

path_origin = '/home/rlops/datasets/AVT/Dongyang/D1/D1.nrrd'
path_origin_mask = '/home/rlops/datasets/AVT/Dongyang/D1/D1.seg.nrrd'

subject_dict = {
    'data': preprocessing(tio.ScalarImage(path_origin)),
    'dense': tio.LabelMap(path_origin_mask),
}

ct_scan_data = subject_dict['data'].data[0]
spacing = subject_dict['data'].spacing
extent = (
    0,
    ct_scan_data.shape[2] * spacing[2],
    0,
    ct_scan_data.shape[1] * spacing[1]
)
mask = subject_dict['dense'].data[0].bool()

out_shape = mask.unsqueeze(0).unsqueeze(0).size()
img_gt = mask.unsqueeze(0).unsqueeze(0).numpy()
dist_transform = np.zeros(out_shape)

for b in range(out_shape[0]): # batch size
    for c in range(out_shape[1]):
        posmask = img_gt[b].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            posdis = distance_transform_edt(posmask)
            negdis = distance_transform_edt(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))*negmask - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))*posmask
            sdf[boundary==1] = 0
            dist_transform[b][c] = sdf
dist_transform = dist_transform[0][0]
#dist_transform = distance_transform_edt(mask)
#dist_transform_inv = -1*distance_transform_edt(~mask)
#dist_transform = dist_transform + dist_transform_inv
#ma = np.maximum(abs(dist_transform.min()),abs(dist_transform.max()))
#rescale = tio.transforms.RescaleIntensity(out_min_max=(-1,1),in_min_max=(-ma,ma))
#dist_transform = torch.tensor(dist_transform).unsqueeze(0)
#dist_transform = rescale(dist_transform)
#dist_transform = dist_transform.squeeze().numpy()

laplacian_kernel = torch.tensor([[[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]],

                                 [[0, 1, 0],
                                  [1, -6, 1],
                                  [0, 1, 0]],

                                 [[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]]], dtype=torch.float32)
displacement_map = torch.tensor(dist_transform, dtype=torch.float32)
padded_displacement_map = F.pad(displacement_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='reflect')
laplacian_map = F.conv3d(padded_displacement_map, laplacian_kernel.unsqueeze(0).unsqueeze(0))
laplacian_map = laplacian_map.squeeze()

# skeleton
#d_mat = torch.abs(laplacian_map)

skel = skeletonize(speed_power=1.2, Euler_step_size=0.5, depth_th=1, length_th=None, simple_path=False, verbose=True)
skeletons = skel.skeleton(mask.numpy())
d_mat = np.zeros(mask.numpy().shape)
for sk in skeletons:
    d_mat[sk.astype(np.int64)[:, 0], sk.astype(np.int64)[:, 1], sk.astype(np.int64)[:, 2]] = 1


sss = tio.LabelMap(tensor=torch.from_numpy(d_mat).unsqueeze(0))
sss.save('spooky_scary_skeleton.nii.gz')
fig, (ax_ct, ax_mask, ax_dist, ax_skl) = plt.subplots(1, 4, figsize=(18, 6))


# Display the initial slice (assuming the slice index is 0)
current_slice = 140
img = ax_ct.imshow(ct_scan_data[current_slice], cmap='gray', extent=extent)
img_mask = ax_mask.imshow(mask[current_slice], cmap='binary', interpolation='nearest', vmin=0, vmax=1, extent=extent)
img_dist = ax_dist.imshow(dist_transform[current_slice], cmap='jet', vmin=dist_transform.min(), vmax=dist_transform.max(), extent=extent)
img_skl = ax_skl.imshow(d_mat[current_slice], cmap='jet', vmin=d_mat.min(), vmax=d_mat.max(), extent=extent)

ax_ct.set_title('CT Scan')
ax_ct.set_xlabel('X')
ax_ct.set_ylabel('Y')

ax_mask.set_title('Mask')
ax_mask.set_xlabel('X')
ax_mask.set_ylabel('Y')

ax_dist.set_title('Distance')
ax_dist.set_xlabel('X')
ax_dist.set_ylabel('Y')

ax_skl.set_title('Laplacian')
ax_skl.set_xlabel('X')
ax_skl.set_ylabel('Y')

# Create a slider widget
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Define the position of the slider
slider = Slider(ax_slider, 'Slice', 0, ct_scan_data.shape[0] - 1, valinit=current_slice, valstep=1)

# Function to update the displayed image when the slider value changes
def update_slice(val):
    current_slice = int(val)
    img.set_data(ct_scan_data[current_slice])
    img_mask.set_data(mask[current_slice])
    img_dist.set_data(dist_transform[current_slice])
    img_skl.set_data(d_mat[current_slice])
    fig.canvas.draw()

# Connect the slider's on_changed event to the update_slice function
slider.on_changed(update_slice)

# Show the plot
plt.show()
bin2dist_disp(path_origin_mask)
print('1')