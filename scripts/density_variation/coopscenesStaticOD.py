import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Für Fortschrittsanzeige
from coopscenes import Dataloader
from sklearn.cluster import KMeans

##############################################
# Teil 1: Erstellen (oder Laden) des Intermediate Models
##############################################

# Parameter
voxel_size = 0.5          # Voxelgröße in Metern
frame_step = 1            # Verarbeite jeden Frame (kann angepasst werden)
group_size = 5            # Gruppengröße: 5 aufeinanderfolgende Frames
cv_threshold = 0.3        # Schwellenwert für den Koeffizienten der Variation (CV)
zero_proportion_threshold = 0.8  # Voxel gilt als "Road user passing area", wenn in >80% der Gruppen kein Punkt vorhanden ist
load_intermediate_model = False  # Falls True, wird das Zwischenmodell geladen, falls vorhanden

# Dynamisch generierter Dateipfad für das Zwischenmodell
intermediate_model_path = f"/home/vsv-cluster/workspace/LidarObjectDetection/intermediate_voxel_model_vs_{voxel_size}_fs_{frame_step}.pkl"

def get_voxel_index(points, voxel_size):
    """Berechnet die Voxelindizes für jeden Punkt, d.h. teilt den Raum in gleich große Würfel ein."""
    return np.floor(points / voxel_size).astype(np.int32)

# Schritt 1: Laden oder Berechnen der voxel_densities
if load_intermediate_model and os.path.exists(intermediate_model_path):
    with open(intermediate_model_path, "rb") as f:
        intermediate_model = pickle.load(f)
    voxel_densities = intermediate_model["voxel_densities"]
    frame_step = intermediate_model["frame_step"]
    total_frames_used = intermediate_model["total_frames_used"]
    print("Intermediate voxel_densities wurden geladen aus '{}'.".format(intermediate_model_path))
else:
    frames_voxel_counts = []  # Für jeden verarbeiteten Frame wird ein Dictionary gespeichert: {Voxelindex: Punktzahl}
    frame_count = 0

    dataset = Dataloader('/mnt/hot_data/dataset/coopscenes/seq_1')
    for record in tqdm(dataset, desc="Records processed"):
        for frame in record:
            if frame_count % frame_step != 0:
                frame_count += 1
                continue

            pcl = frame.tower.lidars.UPPER_PLATFORM
            pts = np.vstack((pcl['x'], pcl['y'], pcl['z'])).T
            voxel_inds = get_voxel_index(pts, voxel_size)
            counts = {}
            for voxel in voxel_inds:
                key = tuple(voxel)
                counts[key] = counts.get(key, 0) + 1
            frames_voxel_counts.append(counts)
            frame_count += 1

    total_frames_used = len(frames_voxel_counts)
    print("Total frames processed (mit Schrittweite {}):".format(frame_step), total_frames_used)

    # Vereinige alle Voxels, die in irgendeinem verarbeiteten Frame belegt waren
    all_voxels = set()
    for counts in frames_voxel_counts:
        all_voxels.update(counts.keys())

    # Erstelle für jedes Voxel die Dichte-Zeitreihe (Punktzahl pro Frame)
    voxel_densities = {}
    for voxel in all_voxels:
        density_series = []
        for counts in frames_voxel_counts:
            density_series.append(counts.get(voxel, 0))
        voxel_densities[voxel] = np.array(density_series)

    # Speichere das Zwischenresultat
    intermediate_model = {
        "voxel_densities": voxel_densities,
        "voxel_size": voxel_size,
        "frame_step": frame_step,
        "total_frames_used": total_frames_used
    }
    with open(intermediate_model_path, "wb") as f:
        pickle.dump(intermediate_model, f)
    print("Intermediate voxel_densities wurden in '{}' gespeichert.".format(intermediate_model_path))


##############################################
# Teil 2: Erstellung des Final Background Models
##############################################

# Ziel: Das finale Modell soll den statischen Teil der Punktwolke beschreiben.
# Dazu aggregieren wir die Zeitreihen in nicht-überlappende Gruppen von group_size Frames (hier 5).
# Anschließend berechnen wir für jedes Voxel:
#   - Mittelwert, Standardabweichung, CV (Standardabweichung / Mittelwert)
#   - Zero-Proportion = (# Gruppen, in denen der aggregierte Wert 0 ist) / (Anzahl der Gruppen)
# Wir klassifizieren die Voxels in 3 Kategorien:
#   1. Road user passing area: Zero-Proportion > zero_proportion_threshold
#   2. No road user passing area: CV < cv_threshold (konstante Dichte) --> statischer Hintergrund
#   3. Marginal area: alle übrigen Voxels
# Für das finale statische Hintergrundmodell nehmen wir alle Voxels aus Kategorie 2.

def aggregate_in_groups_sum(values, group_size=5):
    """
    Aggregiert die Werte in nicht-überlappende Gruppen der Länge group_size,
    indem für jede Gruppe die Summe der Werte berechnet wird.
    """
    values = np.array(values)
    n = len(values)
    num_groups = n // group_size
    if num_groups == 0:
        return values
    aggregated = values[:num_groups * group_size].reshape(num_groups, group_size).sum(axis=1)
    return aggregated

# Aggregiere für jedes Voxel die Dichte-Zeitreihe in Gruppen von group_size Frames
aggregated_voxel_densities = {}
for voxel, densities in voxel_densities.items():
    aggregated_voxel_densities[voxel] = aggregate_in_groups_sum(densities, group_size=group_size)

# Für jedes Voxel berechnen wir statistische Kennzahlen:
# Mittelwert, Standardabweichung, CV, und Zero-Proportion (Anteil Gruppen mit Summe 0)
voxel_stats = {}
num_groups = None
for voxel, agg_values in aggregated_voxel_densities.items():
    agg_values = np.array(agg_values)
    mean_val = np.mean(agg_values)
    std_val = np.std(agg_values)
    cv = std_val / mean_val if mean_val > 0 else 0
    zero_count = np.sum(agg_values == 0)
    num_groups = len(agg_values)  # Alle Voxels haben dieselbe Anzahl Gruppen, wenn sie vollständig sind.
    zero_ratio = zero_count / num_groups if num_groups > 0 else 0
    voxel_stats[voxel] = (mean_val, std_val, cv, zero_ratio)

# Klassifiziere die Voxels in 3 Kategorien:
voxel_categories = {}
for voxel, stats in voxel_stats.items():
    mean_val, std_val, cv, zero_ratio = stats
    if zero_ratio > zero_proportion_threshold:
        voxel_categories[voxel] = 1  # Road user passing area
    elif cv < cv_threshold:
        voxel_categories[voxel] = 2  # No road user passing area (statischer Hintergrund)
    else:
        voxel_categories[voxel] = 3  # Marginal area

# Optional: Zähle die Voxels pro Kategorie
cat_counts = {1: 0, 2: 0, 3: 0}
for cat in voxel_categories.values():
    cat_counts[cat] += 1
print("Voxel-Klassifikation:")
print("  Road user passing area (Kategorie 1):", cat_counts[1])
print("  No road user passing area (Kategorie 2, statisch):", cat_counts[2])
print("  Marginal area (Kategorie 3):", cat_counts[3])

# Finales Hintergrundmodell: Wir nehmen als statischen Hintergrund alle Voxels aus Kategorie 2
static_voxels = [voxel for voxel, cat in voxel_categories.items() if cat == 2]
print("Anzahl statischer Voxels (Final Background Model):", len(static_voxels))

# Speichere das finale Hintergrundmodell
final_model_path = f"/home/vsv-cluster/workspace/LidarObjectDetection/background_model_vs_{voxel_size}_fs_{frame_step}_frames_{total_frames_used}.pkl"
final_model = {
    "static_voxels": static_voxels,
    "voxel_stats": voxel_stats,
    "voxel_categories": voxel_categories,
    "voxel_size": voxel_size,
    "group_size": group_size,
    "total_frames_used": total_frames_used,
    "num_groups": num_groups,
    "cv_threshold": cv_threshold,
    "zero_proportion_threshold": zero_proportion_threshold
}
with open(final_model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Hintergrundmodell (statisch) wurde in '{final_model_path}' gespeichert.")

##############################################
# Hinweis:
# Das finale Modell (background_model) enthält die Liste der statischen Voxels,
# die den statischen Teil der Punktwolke repräsentieren.
# Dieses Modell kann später geladen werden, um neue LiDAR-Daten zu filtern (z. B. indem Punkte,
# die in einem dieser Voxels liegen, als Hintergrund betrachtet und entfernt werden).
##############################################
