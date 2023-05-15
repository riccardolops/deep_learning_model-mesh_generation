import os
from dataloader.AugFactory import AugFactory
import torchio as tio
import yaml
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import distance_transform_edt
matplotlib.use('GTK3Agg')

with open('configs/preprocessing.yaml', 'r') as preproc_file:
    preproc = yaml.load(preproc_file, yaml.FullLoader)
preprocessing = AugFactory(preproc).get_transform()

path_origin = '/home/rlops/datasets/AVT/Dongyang/D1/D1.nrrd'
path_origin_mask = '/home/rlops/datasets/AVT/Dongyang/D1/D1.seg.nrrd'

datafile = os.path.splitext(path_origin)[1]
if datafile == '.nrrd':
    from utils.utils import nrrd_reader as ex_reader
elif datafile == '.nii.gz':
    from utils.utils import nib_reader as ex_reader
subject_dict = {
    'data': preprocessing(tio.ScalarImage(path_origin, reader=ex_reader)),
    'dense': tio.LabelMap(path_origin_mask, reader=ex_reader),
}

ct_scan_data = subject_dict['data'].data[0]
spacing = subject_dict['data'].spacing
extent = (
    0,
    ct_scan_data.shape[2] * spacing[2],
    0,
    ct_scan_data.shape[1] * spacing[1]
)
mask = subject_dict['dense'].data[0]

dist_transform = distance_transform_edt(mask)

fig, (ax_ct, ax_mask, ax_dist) = plt.subplots(1, 3, figsize=(18, 6))


# Display the initial slice (assuming the slice index is 0)
current_slice = 0
img = ax_ct.imshow(ct_scan_data[current_slice], cmap='gray', extent=extent)
img_mask = ax_mask.imshow(mask[current_slice], cmap='binary', interpolation='nearest', vmin=0, vmax=1, extent=extent)
img_dist = ax_dist.imshow(dist_transform[current_slice], cmap='jet', vmin=0.0, vmax=dist_transform.max(), extent=extent)

ax_ct.set_title('CT Scan')
ax_ct.set_xlabel('X')
ax_ct.set_ylabel('Y')

ax_mask.set_title('Mask')
ax_mask.set_xlabel('X')
ax_mask.set_ylabel('Y')

ax_dist.set_title('Distance')
ax_dist.set_xlabel('X')
ax_dist.set_ylabel('Y')

# Create a slider widget
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Define the position of the slider
slider = Slider(ax_slider, 'Slice', 0, ct_scan_data.shape[0] - 1, valinit=current_slice, valstep=1)

# Function to update the displayed image when the slider value changes
def update_slice(val):
    current_slice = int(val)
    img.set_data(ct_scan_data[current_slice])
    img_mask.set_data(mask[current_slice])
    img_dist.set_data(dist_transform[current_slice])
    fig.canvas.draw()

# Connect the slider's on_changed event to the update_slice function
slider.on_changed(update_slice)

# Show the plot
plt.show()

print('1')