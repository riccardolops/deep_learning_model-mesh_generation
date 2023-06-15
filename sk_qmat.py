import pyvista as pv
import numpy as np
import math

def check_arrays(list_of_arrays, target_array):
    for array in list_of_arrays:
        if np.isin(target_array, array).all():
            return True
    return False

filename = '/home/rlops/Downloads/Telegram Desktop/test.ma___v_100___e_117___f_18.ma'
y_model = pv.get_reader('/home/rlops/datasets/AVT/Dongyang/D2/D2_model.ply').read()

with open(filename, 'r') as file:
    lines = file.readlines()

# Extract the number of points, connections, and triangles
num_points, num_connections, num_triangles = map(int, lines[0].split())

points = np.empty((num_points, 3))
radius = np.empty(num_points)

# Extract the points' coordinates and radius
i = 0
for line in lines[1:num_points + 1]:
    data = line.split()
    points[i] = [coord for coord in data[1:4]]
    radius[i] = data[4]
    i = i + 1

# Create the PyVista mesh
mesh = pv.PolyData()
mesh.points=points
mesh.point_data['radius'] = radius



# Extract the connections and triangles
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

np.delete(mesh.faces.reshape(int(len(mesh.faces)/4), 4), 0, axis=1)

copy_list = con_list[:]
triangle_normals = []
is_triangle = np.zeros(len(copy_list))
triangle_idx = np.zeros(len(copy_list))
t = 0
for triangle in triangles:
    vertex_indices = triangle[1:]
    v0, v1, v2 = points[vertex_indices]
    triangle_normal = np.cross(v1 - v0, v2 - v0)
    normalized_triangle_normal = triangle_normal / np.sqrt(sum(triangle_normal**2))
    triangle_normals.append(normalized_triangle_normal)
    flag1 = True
    flag2 = True
    flag3 = True
    i = 0
    for connection in copy_list:
        if np.isin(triangle[1:3], connection).all() and flag1:
            is_triangle[i] = 1
            triangle_idx[i] = t
            if check_arrays(con_list, np.array(connection)):
                con_list.remove(connection)
            flag1 = False
        elif np.isin(triangle[2:4], connection).all() and flag2:
            is_triangle[i] = 1
            triangle_idx[i] = t
            if np.isin(triangle[2:4], con_list).all():
                con_list.remove(connection)
            flag2 = False
        elif np.isin(triangle[1:4:2], connection).all() and flag3:
            is_triangle[i] = 1
            triangle_idx[i] = t
            if np.isin(triangle[1:4:2], con_list).all():
                con_list.remove(connection)
            flag3 = False
        if not flag1 and not flag2 and not flag3:
            break
        i += 1
    t += 1


k = 3
mid_points = np.empty((len(copy_list)*(k-1), 3))
mid_points_radius = np.empty(len(copy_list)*(k-1))
points_normal = np.empty((len(copy_list)*(k-1)*2, 3))
i = 0
j = 0
c = 0
for connection in copy_list:
    p1_i, p2_i = connection
    if is_triangle[c] == 0:
        direction = points[p2_i] - points[p1_i]
        normalized_direction = direction / np.sqrt(sum(direction**2))
        perpendicular_vector = np.array([1., 1., (-normalized_direction[0]-normalized_direction[1])/normalized_direction[2]])
        normalized_perpendicular = perpendicular_vector / np.sqrt(sum(perpendicular_vector**2))
    elif is_triangle[c] == 1:
        normalized_perpendicular = triangle_normals[int(triangle_idx[c])]

    for p in range(k-1):
        mid_points[i] = points[p1_i] + ((points[p2_i] - points[p1_i]) * (p+1) / k)
        mid_points_radius[i] = radius[p1_i] + ((radius[p2_i] - radius[p1_i]) * (p+1) / k)

        point_positive = mid_points[i] + (normalized_perpendicular * mid_points_radius[i])
        point_negative = mid_points[i] - (normalized_perpendicular * mid_points_radius[i])
        points_normal[j] = point_positive
        points_normal[j + 1] = point_negative
        j += 2
        i += 1
    c += 1


# Display the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='radius')
for point, rad in zip(mid_points, mid_points_radius):
    plotter.add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.10)
for point, rad in zip(points, radius):
    plotter.add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.10)
plotter.add_points(mid_points)
plotter.add_points(points)
plotter.add_points(points_normal)
plotter.add_mesh(y_model, opacity=0.10, color='#FF0000')
plotter.show()
