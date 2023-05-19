import torchio as tio
import torch
import nibabel as nib
import nrrd
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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
def bin2dist_disp(path):
    mask = tio.LabelMap(path)
    dist_transform_unsampled = distance_transform_edt(mask.data)
    dist_transform = distance_transform_edt(mask.data, sampling=[abs(mask.affine[3][3]), abs(mask.affine[0][0]), abs(mask.affine[1][1]), abs(mask.affine[2][2])])
    dist = tio.ScalarImage(tensor=dist_transform, affine=mask.affine)
    dist_unsa = tio.ScalarImage(tensor=dist_transform_unsampled, affine=mask.affine)
    spacing = mask.spacing
    extent = (
        0,
        dist_transform[0].shape[2] * spacing[2],
        0,
        dist_transform[0].shape[1] * spacing[1]
    )
    fig, (ax_mask, ax_dist, ax_dist_unsa) = plt.subplots(1, 3, figsize=(18, 6))

    current_slice = 0
    img_mask = ax_mask.imshow(mask.data[0][current_slice], cmap='binary', interpolation='nearest', vmin=0, vmax=1,
                              extent=extent)
    img_dist = ax_dist.imshow(dist.data[0][current_slice], cmap='jet', vmin=dist_transform.min(), vmax=dist_transform.max(),
                              extent=extent)
    img_dist_unsa = ax_dist_unsa.imshow(dist_unsa.data[0][current_slice], cmap='jet', vmin=dist_transform_unsampled.min(),
                              vmax=dist_transform_unsampled.max(),
                              extent=extent)
    ax_mask.set_title('Mask')
    ax_mask.set_xlabel('X')
    ax_mask.set_ylabel('Y')

    ax_dist.set_title('Distance')
    ax_dist.set_xlabel('X')
    ax_dist.set_ylabel('Y')

    ax_dist_unsa.set_title('Distance Unsampled')
    ax_dist_unsa.set_xlabel('X')
    ax_dist_unsa.set_ylabel('Y')

    # Create a slider widget
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Define the position of the slider
    slider = Slider(ax_slider, 'Slice', 0, dist_transform[0].shape[0] - 1, valinit=current_slice, valstep=1)

    # Function to update the displayed image when the slider value changes
    def update_slice(val):
        current_slice = int(val)
        img_mask.set_data(mask.data[0][current_slice])
        img_dist.set_data(dist.data[0][current_slice])
        img_dist_unsa.set_data(dist_unsa.data[0][current_slice])
        fig.canvas.draw()

    # Connect the slider's on_changed event to the update_slice function
    slider.on_changed(update_slice)

    # Show the plot
    plt.show()

    return dist_transform