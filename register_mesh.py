import torch
from ma2mesh import ma2mesh, skl2norm_p
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
import torchio as tio
from utils.utils import bin2dist1
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

matplotlib.use('GTK3Agg')


x_model = pv.get_reader('/home/rlops/datasets/AVT/Dongyang/D2/D2_model.ply').read()
y_model = pv.get_reader('/home/rlops/datasets/AVT/Dongyang/D8/D8_model.ply').read()

filename = '/home/rlops/Downloads/Telegram Desktop/test.ma___v_100___e_117___f_18.ma'

skl_mesh = ma2mesh(filename)

learningRate = 10
epochs = 401
checkpoint_period = 200
epsilon = 5
step_size = 300
n_points = len(skl_mesh.points)
checkpoints = []
writer = SummaryWriter('/home/rlops/deep_learning_model-mesh_generation/skl', flush_secs=2)


plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("D2", font_size=30)
plotter.add_mesh(skl_mesh)
plotter.add_mesh(x_model, opacity=0.10)

plotter.subplot(0, 1)
plotter.add_text("D8", font_size=30)
plotter.add_mesh(y_model, opacity=0.10)
plotter.show()

constant_displacement = torch.randn(3, requires_grad=True)
offset_const = torch.tensor([0., 0., 0.], requires_grad=True)
optimizer_pre = torch.optim.SGD([offset_const], lr=0.1)
offset_mesh = skl_mesh
for pre in tqdm(range(60)):
    optimizer_pre.zero_grad()
    output = torch.from_numpy(offset_mesh.points).unsqueeze(0) + offset_const.repeat(n_points, 1)
    loss, _ = chamfer_distance(output.float(), torch.from_numpy(y_model.points).unsqueeze(0).float())
    loss.backward()
    optimizer_pre.step()
    print('epoch {}, loss {}'.format(pre, loss.item()))

offset_mesh.points += offset_const.repeat(n_points, 1).detach().numpy()
x_model.points += offset_const.repeat(x_model.n_points, 1).detach().numpy()

weight = 0.00005

plotter = pv.Plotter(notebook=False)
plotter.add_text("X", font_size=30, color='#00FF00')
plotter.add_mesh(offset_mesh, color='#00FF00')
plotter.add_mesh(x_model, opacity=0.10, color='#00FF00')
plotter.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
plotter.add_mesh(y_model, opacity=0.10, color='#FF0000')
camera = plotter.camera
camera.SetFocalPoint(y_model.center_of_mass())
camera.Azimuth(160)
camera.Elevation(-30)
camera.zoom(0.9)
#plotter.open_gif("reg_p.gif")
#plotter.write_frame()

offset_parameters = torch.randn(n_points, 4, requires_grad=True)
optimizer = torch.optim.SGD([offset_parameters], lr=learningRate)
scheduler1 = ExponentialLR(optimizer, gamma=0.99999)
scheduler2 = MultiStepLR(optimizer, milestones=[100, 120, 130, 150, 160], gamma=0.5)

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    points_out, loss_dis = skl2norm_p(offset_mesh, offset_parameters)
    loss_chamfer, _ = chamfer_distance(points_out.unsqueeze(0).float(), torch.from_numpy(y_model.points).unsqueeze(0).float())
    loss_distance = torch.sum(torch.sqrt(torch.sum(offset_parameters ** 2, dim=1)))
    loss = loss_chamfer + weight * loss_distance

    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('loss', loss, epoch)
    loss.backward()
    optimizer.step()
    #scheduler.step(loss)
    scheduler1.step()
    scheduler2.step()

    if epoch % checkpoint_period == 0:
        skl_mesh.points = offset_mesh.points + offset_parameters[:, :3].detach().numpy()
        print('LR: {}'.format(optimizer.param_groups[0]['lr']))
        #checkpoints.append(output.points_packed().detach().numpy().astype(np.float32))
        # plot_pointcloud(output.points_packed().detach(), 'epoch {}'.format(epoch))
        plotter = pv.Plotter(notebook=False)
        #plotter.add_volume(dist.data.squeeze().numpy(), cmap="bone")
        plotter.add_text("X", font_size=30, color='#00FF00')
        plotter.add_mesh(skl_mesh, color='#00FF00')
        plotter.add_points(points_out.detach().numpy(), color='#00FF00')
        plotter.add_mesh(x_model, opacity=0.10, color='#00FF00')
        plotter.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
        plotter.add_mesh(y_model, opacity=0.10, color='#FF0000')
        #plotter.write_frame()
        plotter.show()
        plotter.clear()

    print('epoch {}, loss_chamfer {}'.format(epoch, loss_chamfer.item()))
    print('epoch {}, loss_distance {}'.format(epoch, loss_distance.item()))
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print('finish')
plotter.close()
writer.close()
