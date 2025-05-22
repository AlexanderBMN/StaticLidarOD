import pickle
import numpy as np
import matplotlib.pyplot as plt

# Pfad zum gespeicherten Zwischenmodell (anpassen, falls nötig)
intermediate_model_path = "/home/vsv-cluster/workspace/LidarObjectDetection/intermediate_voxel_model_vs_0.5_fs_10.pkl"

# Lade das Zwischenmodell
with open(intermediate_model_path, "rb") as f:
    intermediate_model = pickle.load(f)

# Originale voxel_densities: Dictionary: Voxel -> Zeitreihe (Punktzahl pro Frame)
voxel_densities = intermediate_model["voxel_densities"]
voxel_size = intermediate_model["voxel_size"]

# Für jeden Voxel:
# - Berechne die aggregierte Punktzahl über alle Frames
# - Bestimme den Mittelpunkt des Voxels und die Entfernung vom Ursprung
voxel_total_counts = []
voxel_distances = []

for voxel, densities in voxel_densities.items():
    total_count = np.sum(densities)
    # Mittelpunkt des Voxels (angenommen, der Voxel geht von i*voxel_size bis (i+1)*voxel_size)
    center = (np.array(voxel) + 0.5) * voxel_size
    distance = np.linalg.norm(center)
    voxel_total_counts.append(total_count)
    voxel_distances.append(distance)

voxel_total_counts = np.array(voxel_total_counts)
voxel_distances = np.array(voxel_distances)

# Nur Voxels mit einer Entfernung von bis zu 150 m berücksichtigen
max_distance = 150
mask = voxel_distances <= max_distance
filtered_distances = voxel_distances[mask]
filtered_total_counts = voxel_total_counts[mask]

# Erstelle ein Histogramm der aggregierten Punktzahlen in Abhängigkeit von der Entfernung
num_bins = 50
bins = np.linspace(0, max_distance, num_bins)
hist, bin_edges = np.histogram(filtered_distances, bins=bins, weights=filtered_total_counts)

# Berechne die Bin-Mitte für die x-Achse
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), color='steelblue', edgecolor='black')
plt.xlabel("Entfernung (m)")
plt.ylabel("Anzahl Punkte (aggregiert, über alle Frames)")
plt.title("Verteilung der Punktzahlen bis 150m Entfernung")
plt.grid(True)
plt.show()
