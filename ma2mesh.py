import numpy as np
import pyvista as pv
import torch
import math

def get_rotation_matrix(axis, angle_rad):
    """
    Returns a 3x3 rotation matrix for rotating a vector around the given axis by the specified angle.
    """
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    cross_product_matrix = torch.tensor([[0, -axis[2], axis[1]],
                                         [axis[2], 0, -axis[0]],
                                         [-axis[1], axis[0], 0]])
    rotation_matrix = torch.eye(3) * cos_theta + (1 - cos_theta) * torch.outer(axis, axis) + sin_theta * cross_product_matrix
    return rotation_matrix

def ma2mesh(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    num_points, num_connections, num_triangles = map(int, lines[0].split())
    points = np.empty((num_points, 3))
    radius = np.empty(num_points)

    i = 0
    for line in lines[1:num_points + 1]:
        data = line.split()
        points[i] = [coord for coord in data[1:4]]
        radius[i] = data[4]
        i += 1

    mesh = pv.PolyData()
    mesh.points = points
    mesh.point_data['radius'] = radius

    connections = []
    con_list = []
    triangles = []
    for line in lines[num_points + 1:]:
        data = line.split()
        if data[0] == 'e':
            num_points_in_connection = 2
            connection = list(map(int, data[1:]))
            con_list.append(connection)
            connections.append([num_points_in_connection] + connection)
        elif data[0] == 'f':
            num_points_in_triangle = 3
            triangle = list(map(int, data[1:]))
            triangles.append([num_points_in_triangle] + triangle)

    # Add the connections and triangles to the mesh
    mesh.lines = connections
    mesh.faces = triangles
    return mesh

def skl2norm_p(mesh, offset_points):
    points_before = torch.from_numpy(mesh.points).float()
    points = torch.from_numpy(mesh.points).float() + offset_points[:, :3]
    radius = torch.from_numpy(mesh.point_data['radius']).float() + offset_points[:, 3:4].squeeze()
    np_triangles = np.delete(mesh.faces.reshape(int(len(mesh.faces)/4), 4), 0, axis=1)
    np_connections = np.delete(mesh.lines.reshape(int(len(mesh.lines)/3), 3), 0, axis=1)
    triangle_normals = []
    triangle_idx = np.ones(len(np_connections)).astype(int)*-1
    t = 0
    for triangle in np_triangles:
        v0, v1, v2 = points[triangle]
        triangle_normal = torch.cross(v1 - v0, v2 - v0)
        normalized_triangle_normal = triangle_normal / torch.norm(triangle_normal)
        triangle_normals.append(normalized_triangle_normal)
        flag1 = True
        flag2 = True
        flag3 = True
        i = 0
        for connection in np_connections:
            if np.isin(triangle[0:2], connection).all() and flag1:
                triangle_idx[i] = t
                flag1 = False
            elif np.isin(triangle[1:3], connection).all() and flag2:
                triangle_idx[i] = t
                flag2 = False
            elif np.isin(triangle[0:3:2], connection).all() and flag3:
                triangle_idx[i] = t
                flag3 = False
            if not flag1 and not flag2 and not flag3:
                break
            i += 1
        t += 1

    k = 3 # mid_point per connection = k - 1
    normals_per_mid_point = 6
    con_tri = sum(triangle_idx != -1)
    con_no_tri = len(np_connections)-con_tri
    mid_points_con = torch.empty((con_no_tri * (k + 1) * normals_per_mid_point, 3))
    mid_points_con_radius = torch.empty((con_no_tri * (k + 1) * normals_per_mid_point, 1))
    normalized_perpendicular_con = torch.empty((con_no_tri * (k + 1) * normals_per_mid_point, 3))
    mid_points_tri = torch.empty((con_tri * (k + 1), 3))
    mid_points_tri_radius = torch.empty((con_tri * (k + 1), 1))
    normalized_perpendicular_tri = torch.empty((con_tri * (k + 1), 3))
    mt = 0
    c = 0
    mc = 0
    ppc = 0
    normals_per_point = 6
    theta = 2 * math.pi/normals_per_point
    loss_distance = torch.zeros((1, 1))
    for connection in np_connections:
        p1_i, p2_i = connection
        distance_before = torch.sqrt(torch.sum((points_before[p1_i] - points_before[p2_i])**2))
        distance_after = torch.sqrt(torch.sum((points[p1_i] - points[p2_i]) ** 2))
        loss_distance += torch.sqrt((distance_after - distance_before)**2)

        radius_p1 = radius[p1_i]
        radius_p2 = radius[p2_i]

        if triangle_idx[c] == -1:
            direction = points[p2_i] - points[p1_i]
            normalized_direction = direction / torch.norm(direction)
            perpendicular_vector = (torch.tensor([0., 0., 1.]) * (-normalized_direction[0] - normalized_direction[1]) / normalized_direction[2]) + torch.tensor([1., 1., 0.])
            perpendicular = perpendicular_vector / torch.norm(perpendicular_vector)
            for pc in range(normals_per_point):
                mid_points_con[ppc] = points[p1_i]
                mid_points_con_radius[ppc] = radius_p1
                angle = theta * pc
                rotation_matrix = get_rotation_matrix(normalized_direction, angle)
                normalized_perpendicular_con[ppc] = torch.matmul(rotation_matrix, perpendicular)
                ppc += 1
            for p in range(k - 1):
                for pc in range(normals_per_point):
                    mid_points_con[ppc] = points[p1_i] + ((points[p2_i] - points[p1_i]) * (p + 1) / k)
                    mid_points_con_radius[ppc] = radius_p1 + ((radius_p2 - radius_p1) * (p + 1) / k)
                    angle = theta * pc
                    rotation_matrix = get_rotation_matrix(normalized_direction, angle)
                    normalized_perpendicular_con[ppc] = torch.matmul(rotation_matrix, perpendicular)
                    ppc += 1
            for pc in range(normals_per_point):
                mid_points_con[ppc] = points[p2_i]
                mid_points_con_radius[ppc] = radius_p2
                angle = theta * pc
                rotation_matrix = get_rotation_matrix(normalized_direction, angle)
                normalized_perpendicular_con[ppc] = torch.matmul(rotation_matrix, perpendicular)
                ppc += 1

        elif triangle_idx[c] != -1:
            perpendicular = triangle_normals[int(triangle_idx[c])]
            mid_points_tri[mt] = points[p1_i]
            mid_points_tri_radius[mt] = radius_p1
            normalized_perpendicular_tri[mt] = perpendicular
            mt += 1
            for p in range(k - 1):
                mid_points_tri[mt] = points[p1_i] + ((points[p2_i] - points[p1_i]) * (p + 1) / k)
                mid_points_tri_radius[mt] = radius_p1 + ((radius_p2 - radius_p1) * (p + 1) / k)
                normalized_perpendicular_tri[mt] = perpendicular
                mt += 1
            mid_points_tri[mt] = points[p2_i]
            mid_points_tri_radius[mt] = radius_p2
            normalized_perpendicular_tri[mt] = perpendicular
            mt += 1
        c += 1

    point_positive_tri = mid_points_tri + (normalized_perpendicular_tri * mid_points_tri_radius.repeat(1, 3))
    point_negative_tri = mid_points_tri - (normalized_perpendicular_tri * mid_points_tri_radius.repeat(1, 3))
    point_seg = mid_points_con + (normalized_perpendicular_con * mid_points_con_radius.repeat(1, 3))
    points_normal = torch.cat((point_positive_tri, point_negative_tri, point_seg), dim=0)
    return points_normal, loss_distance