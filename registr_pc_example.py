import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tensorboardX import SummaryWriter
import numpy as np
import pyvista as pv
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pytorch3d.structures.pointclouds import Pointclouds
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import threading
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

matplotlib.use('GTK3Agg')

def plot_pointcloud(points, title=""):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)

    min_val = min(min(x), min(-y), min(z))
    max_val = max(max(x), max(-y), max(z))

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])

    plt.show()

x_cloud_np = np.load('/home/rlops/datasets/AVT/Dongyang/D2/D2_skl.npy')
x_model = pv.get_reader('/home/rlops/datasets/AVT/Dongyang/D2/D2_model.ply').read()
y_cloud_np = np.load('/home/rlops/datasets/AVT/Dongyang/D8/D8_skl.npy')
y_model = pv.get_reader('/home/rlops/datasets/AVT/Dongyang/D8/D8_model.ply').read()



inputDim = x_cloud_np.shape[0]
outputDim = y_cloud_np.shape[0]
# make poinclouds same N of points
if inputDim < outputDim:
    y_cloud_np = y_cloud_np[:inputDim]
elif outputDim < inputDim:
    x_cloud_np = x_cloud_np[:outputDim]

learningRate = 10
epochs = 401
checkpoint_period = 100
epsilon = 5
step_size = 300
n_points = x_cloud_np.shape[0]
checkpoints = []
writer = SummaryWriter('/home/rlops/deep_learning_model-mesh_generation/skl', flush_secs=2)

x_cloud = Pointclouds(torch.from_numpy(x_cloud_np).unsqueeze(0))
y_cloud = Pointclouds(torch.from_numpy(y_cloud_np).to(torch.float32).unsqueeze(0))

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("D2", font_size=30)
plotter.add_points(x_cloud.points_packed().detach().numpy().astype(np.float32))
plotter.add_mesh(x_model, opacity=0.10)

plotter.subplot(0, 1)
plotter.add_text("D8", font_size=30)
plotter.add_points(y_cloud.points_packed().detach().numpy().astype(np.float32))
plotter.add_mesh(y_model, opacity=0.10)
plotter.show()

offset_parameters = torch.randn(n_points, 3, requires_grad=True)
constant_displacement = torch.randn(3, requires_grad=True)

# Initialize the optimizer with the desired parameters
optimizer = torch.optim.SGD([offset_parameters], lr=learningRate)
# Initialize the scheduler with the desired parameters
#scheduler = ReduceLROnPlateau(optimizer)
scheduler1 = ExponentialLR(optimizer, gamma=0.99999)
scheduler2 = MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.5)

offset_const = torch.tensor([0., 0., 0.], requires_grad=True)
optimizer_pre = torch.optim.SGD([offset_const], lr=0.1)
for pre in tqdm(range(60)):
    optimizer_pre.zero_grad()
    output = x_cloud.offset(offset_const.repeat(n_points, 1))
    loss, _ = chamfer_distance(output.points_packed().unsqueeze(0).float(), y_cloud.points_packed().unsqueeze(0).float())
    loss.backward()
    optimizer_pre.step()
    print('epoch {}, loss {}'.format(pre, loss.item()))

x_cloud = x_cloud.offset(offset_const.repeat(n_points, 1).detach())
x_model.points += offset_const.repeat(x_model.n_points, 1).detach().numpy()
pv.plot(x_cloud.points_packed().detach().numpy().astype(np.float32))

weight = 0.00005

plotter = pv.Plotter(notebook=False)
plotter.add_text("X", font_size=30, color='#00FF00')
plotter.add_points(x_cloud.points_packed().detach().numpy().astype(np.float32), color='#00FF00')
plotter.add_mesh(x_model, opacity=0.10, color='#00FF00')
plotter.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
plotter.add_points(y_cloud.points_packed().detach().numpy().astype(np.float32), color='#FF0000')
plotter.add_mesh(y_model, opacity=0.10, color='#FF0000')
camera = plotter.camera
camera.SetFocalPoint(y_model.center_of_mass())
camera.Azimuth(160)
camera.Elevation(-30)
camera.zoom(0.9)
#plotter.open_gif("reg_p.gif")
#plotter.write_frame()

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    output = x_cloud.offset(offset_parameters)
    loss_chamfer, _ = chamfer_distance(output.points_packed().unsqueeze(0).float(), y_cloud.points_packed().unsqueeze(0).float())
    loss_distance = torch.sum(torch.sqrt(torch.sum(offset_parameters ** 2, dim=1)))
    loss = loss_chamfer + weight * loss_distance
    if epoch % checkpoint_period == 0:
        print('LR: {}'.format(optimizer.param_groups[0]['lr']))
        checkpoints.append(output.points_packed().detach().numpy().astype(np.float32))
        # plot_pointcloud(output.points_packed().detach(), 'epoch {}'.format(epoch))
        plotter = pv.Plotter(notebook=False)
        plotter.add_text("X", font_size=30, color='#00FF00')
        plotter.add_points(output.points_packed().detach().numpy().astype(np.float32), color='#00FF00')
        plotter.add_mesh(x_model, opacity=0.10, color='#00FF00')
        plotter.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
        plotter.add_mesh(y_model, opacity=0.10, color='#FF0000')
        #plotter.write_frame()
        plotter.show()
        plotter.clear()

    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('loss', loss, epoch)
    loss.backward()
    optimizer.step()
    #scheduler.step(loss)
    scheduler1.step()
    scheduler2.step()
    print('epoch {}, loss_chamfer {}'.format(epoch, loss_chamfer.item()))
    print('epoch {}, loss_distance {}'.format(epoch, loss_distance.item()))
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print('finish')
plotter.close()
writer.close()
