import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===============================================
# PARAMETER
# ===============================================
# Pfad zum gespeicherten Zwischenmodell (anpassen, falls nötig)
intermediate_model_path = "/home/vsv-cluster/workspace/LidarObjectDetection/intermediate_voxel_model_vs_0.5_fs_10.pkl"
k = 2  # Anzahl der Cluster für K-means++
max_distance = 100.0  # nur Voxels mit Mittelpunkt ≤ 100 m berücksichtigen

# ===============================================
# Laden des Zwischenmodells
# ===============================================
with open(intermediate_model_path, "rb") as f:
    intermediate_model = pickle.load(f)

# Originale voxel_densities: Dictionary, in dem jedem Voxel (z.B. (i,j,k)) eine Zeitreihe (Punktzahl pro Frame) zugeordnet ist
voxel_densities = intermediate_model["voxel_densities"]
voxel_size = intermediate_model["voxel_size"]

# ===============================================
# Schritt 1: Aggregation in Gruppen von 5 Frames
# ===============================================
def aggregate_in_groups_sum(values, group_size=5):
    """
    Aggregiert die Werte in nicht-überlappende Gruppen der Länge group_size,
    indem für jede Gruppe die Summe der Werte berechnet wird.
    """
    values = np.array(values)
    n = len(values)
    num_groups = n // group_size  # nur volle Gruppen berücksichtigen
    if num_groups == 0:
        return values  # falls weniger als group_size Werte vorhanden sind
    aggregated = values[:num_groups * group_size].reshape(num_groups, group_size).sum(axis=1)
    return aggregated

aggregated_voxel_densities = {}
for voxel, densities in voxel_densities.items():
    aggregated_voxel_densities[voxel] = aggregate_in_groups_sum(densities, group_size=5)

# ===============================================
# Schritt 2: Filterung der Voxels nach Entfernung und Zero-Proportion
# ===============================================
# Für jeden Voxel:
# - Berechne den Mittelpunkt: (voxel + 0.5) * voxel_size
# - Falls die Entfernung > max_distance, wird der Voxel verworfen.
# - Berechne die Zero-Proportion:
#   d_i^4 = (# Gruppen, in denen die aggregierte Summe 0 ist) / (Anzahl der Gruppen)
# - Entferne alle Voxels, bei denen diese Zero-Proportion gleich 1 ist.
filtered_aggregated_voxel_densities = {}
for voxel, agg_values in aggregated_voxel_densities.items():
    # Mittelpunkt berechnen:
    center = (np.array(voxel) + 0.5) * voxel_size
    distance = np.linalg.norm(center)
    if distance > max_distance:
        continue
    # Zero-Proportion berechnen:
    agg_values = np.array(agg_values)
    zero_ratio = np.sum(agg_values == 0) / len(agg_values)
    if zero_ratio < 1.0:  # nur Voxels behalten, die nicht in allen Gruppen leer sind
        filtered_aggregated_voxel_densities[voxel] = agg_values

print("Anzahl Voxels vor Filterung:", len(aggregated_voxel_densities))
print("Anzahl Voxels nach Filterung (Entfernung ≤ 100 m und Zero-Proportion < 1):",
      len(filtered_aggregated_voxel_densities))

# ===============================================
# Schritt 3: Berechnung der Zero-Proportion für jedes gefilterte Voxel
# ===============================================
# d_i^4 = (# Gruppen, in denen die aggregierte Summe 0 ist) / (Anzahl der Gruppen)
zero_props = []
voxels = []
for voxel, agg_values in filtered_aggregated_voxel_densities.items():
    agg_values = np.array(agg_values)
    zero_ratio = np.sum(agg_values == 0) / len(agg_values)
    zero_props.append(zero_ratio)
    voxels.append(voxel)

zero_props = np.array(zero_props).reshape(-1, 1)

# ===============================================
# Schritt 4: K-means++ Clustering auf die Zero-Proportion
# ===============================================
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(zero_props)
print("Cluster Centers (Zero-Proportion):", kmeans.cluster_centers_)

# ===============================================
# Schritt 5: Visualisierung der Zero-Proportion-Verteilung
# ===============================================
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_indices = np.where(clusters == i)[0]
    plt.scatter(cluster_indices, zero_props[cluster_indices], s=5, label=f"Cluster {i}")
plt.xlabel("Voxel Index")
plt.ylabel("Zero Proportion")
plt.title("Zero Proportion Distribution with K-means++ Clustering\n(Voxels ≤ 100m und Zero-Proportion < 1)")
plt.legend()
plt.grid(True)
plt.show()
