import open3d as o3d
import numpy as np
import struct
import os
import matplotlib.colors as mcolors


def read_bin_file(filename, typeLiDAR):
    points = []
    velocities = []
    intensities = []

    with open(filename, "rb") as file:
        while True:
            if typeLiDAR == "Aeva" and int(filename.split("/")[-1].split('.')[0]) > 1691936557946849179:
                data = file.read(29)
                if len(data) < 29:
                    break
                x, y, z, reflectivity, velocity = struct.unpack('fffff', data[:20])
                time_offset_ns = struct.unpack('I', data[20:24])
                line_index = struct.unpack('B', data[24:25])
                intensity = struct.unpack('f', data[25:29])
            elif typeLiDAR == "Aeva" and int(filename.split("/")[-1].split('.')[0]) <= 1691936557946849179:
                data = file.read(25)
                if len(data) < 25:
                    break
                x, y, z, reflectivity, velocity = struct.unpack('fffff', data[:20])
                time_offset_ns = struct.unpack('I', data[20:24])
                line_index = struct.unpack('B', data[24:25])
                intensity = reflectivity  # Falls keine separate Intensität vorhanden ist
            else:
                raise ValueError("Unsupported LiDAR type")

            points.append([x, y, z])
            velocities.append(velocity)
            intensities.append(intensity)

    return np.array(points), np.array(velocities), np.array(intensities)


def velocity_to_rgb_color(velocities):
    # Dynamische Skala: -10 m/s (Blau), 0 m/s (Weiß), +10 m/s (Rot)
    min_v, max_v = -10, 10  # Feste Grenzen

    norm_vel = np.clip(velocities, min_v, max_v)  # Begrenzen auf [-10, 10]
    norm_vel = (norm_vel - min_v) / (max_v - min_v)  # Skaliere auf [0,1]

    # Interpolation zwischen Blau (negativ), Weiß (0) und Rot (positiv)
    colors = np.zeros((len(velocities), 3))
    colors[:, 0] = np.clip(2 * norm_vel, 0, 1)  # Rot-Kanal
    colors[:, 2] = np.clip(2 * (1 - norm_vel), 0, 1)  # Blau-Kanal
    colors[:, 1] = np.clip(2 * (0.5 - np.abs(norm_vel - 0.5)), 0, 1)  # Grün für Weiß-Balance

    return colors


def visualize_point_cloud(points, velocities):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Geschwindigkeit basierte Farbgebung
    colors = velocity_to_rgb_color(velocities)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Open3D Visualisierung mit schwarzem Hintergrund
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Setzt den Hintergrund auf Schwarz
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


# Hardcoded values
filename = "/home/vsv-cluster/dataset/sample/Aeva/1689519071059358083.bin"
typeLiDAR = "Aeva"

points, velocities, intensities = read_bin_file(filename, typeLiDAR)
visualize_point_cloud(points, velocities)