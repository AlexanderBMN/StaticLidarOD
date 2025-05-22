import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Pfad zur Punktwolke
file_path = "/mnt/hot_data/helimos/Aeva/velodyne/000000.bin"

# Laden der Binärdatei als float32
points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# Extrahiere x, y, z und Intensität
xyz = points[:, :3]
intensity = points[:, 3]  # Intensität

# Erstelle eine Open3D Punktwolke
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Intensität normalisieren für die Farbcodierung
intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
colors = plt.cm.jet(intensity_normalized)[:, :3]  # Jet-Colormap für Intensität

# Farben zuweisen
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualisierung starten
o3d.visualization.draw_geometries([pcd], window_name="Interaktive Punktwolke (Farbe = Intensität)")
