#!/usr/bin/env python3
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from coopscenes import Dataloader
from scipy.spatial import ConvexHull


# =============================================================================
# 1. Hilfsfunktionen (Thresholding & Histogramm)
# =============================================================================
def thresholding(distances, dist_max=200, fine_bin_num=200):
    """
    Bestimmt einen Schwellwert (Threshold) für die Hintergrund-/Vordergrundtrennung
    mittels Coarse-Fine Triangle Algorithm (CFTA).
    """
    original_frame_size = len(distances)
    non_zero_distances = distances[distances > 0]
    if len(non_zero_distances) == 0:
        return dist_max
    if len(non_zero_distances) < original_frame_size * 0.2:
        return np.max(non_zero_distances)
    distances = non_zero_distances

    n_bins = max(int(round(np.max(distances))), 1)
    N_coarse, edges_coarse = np.histogram(distances, bins=n_bins)
    max_bin_idx = np.argmax(N_coarse)
    if max_bin_idx != len(edges_coarse) - 2:
        coarse_bin_interval = edges_coarse[1] - edges_coarse[0]
        dist_limit = edges_coarse[max_bin_idx] + coarse_bin_interval * 5
        distances = distances[distances <= dist_limit]
    if len(distances) < 2:
        return dist_max

    max_range = np.max(distances)
    bin_half = round(max_range / fine_bin_num, 2)
    if bin_half == 0:
        bin_half = 0.01
    step = bin_half * 2
    bins = np.arange(-bin_half, max_range + step, step)
    hist, edges = np.histogram(distances, bins=bins)
    hist = hist + 1
    lehisto = hist

    xmax = int(np.round(np.argmax(hist)))
    f_bin = 0
    if xmax <= 1:
        return 0
    ptx0, pty0 = f_bin, lehisto[f_bin]
    ptx1, pty1 = xmax - 1, lehisto[xmax]
    k = (pty0 - pty1) / (ptx0 - ptx1 + 1e-6)
    a, b, c = k, -1, pty1
    x_indices = np.arange(f_bin, xmax)
    y_values = lehisto[x_indices] - 1
    L = np.abs((a * x_indices + b * y_values + c) / np.sqrt(a ** 2 + b ** 2))
    max_L_idx = np.where(L == np.max(L))[0]
    if len(max_L_idx) == 0:
        return dist_max
    level_idx = int(np.round(np.mean(max_L_idx)))
    threshold = edges[level_idx] + bin_half
    return threshold


def visualize_CFTA(distances, fine_bin_num=200, dist_max=200):
    """
    Visualisiert den CFTA-Ansatz zur Ermittlung des Schwellenwerts.
    """
    original_frame_size = len(distances)
    full_data = distances[distances > 0]
    if len(full_data) == 0:
        print("Keine gültigen Werte vorhanden.")
        return dist_max
    if len(full_data) < original_frame_size * 0.2:
        threshold = np.max(full_data)
        print("Weniger als 20% gültige Werte, nutze Maximum:", threshold)
        return threshold

    data = full_data.copy()
    n_bins = max(int(round(np.max(data))), 1)
    N_coarse, edges_coarse = np.histogram(data, bins=n_bins)
    max_bin_idx = np.argmax(N_coarse)
    if max_bin_idx != len(edges_coarse) - 2:
        coarse_bin_interval = edges_coarse[1] - edges_coarse[0]
        dist_limit = edges_coarse[max_bin_idx] + coarse_bin_interval * 5
        data = data[data <= dist_limit]
    if len(data) < 2:
        return dist_max

    max_range = np.max(data)
    bin_half = round(max_range / fine_bin_num, 2)
    if bin_half == 0:
        bin_half = 0.01
    step = bin_half * 2
    bins = np.arange(-bin_half, max_range + step, step)
    hist, edges = np.histogram(data, bins=bins)
    hist = hist + 1
    centers = (edges[:-1] + edges[1:]) / 2

    xmax = int(np.round(np.argmax(hist)))
    f_bin = 0
    if xmax <= 1:
        print("Histogramm zu schmal, kein sinnvoller Schwellenwert ermittelbar.")
        return 0
    point1 = (centers[f_bin], hist[f_bin])
    point2 = (centers[xmax - 1], hist[xmax])
    k = (point1[1] - point2[1]) / (point1[0] - point2[0] + 1e-6)
    x_vals = centers[f_bin:xmax]
    line_y = k * (x_vals - point1[0]) + point1[1]
    L = np.abs(hist[f_bin:xmax] - line_y) / np.sqrt(1 + k ** 2)
    max_idx_relative = np.argmax(L)
    max_idx = f_bin + max_idx_relative
    threshold = edges[max_idx] + bin_half

    full_max_range = np.max(full_data)
    bin_half_full = round(full_max_range / fine_bin_num, 2)
    if bin_half_full == 0:
        bin_half_full = 0.01
    step_full = bin_half_full * 2
    bins_full = np.arange(-bin_half_full, full_max_range + step_full, step_full)
    hist_full, edges_full = np.histogram(full_data, bins=bins_full)
    hist_full = hist_full + 1
    centers_full = (edges_full[:-1] + edges_full[1:]) / 2

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].bar(centers, hist, width=step * 0.9, color='lightgray', label='Histogramm (getrimmt)')
    axs[0].plot(x_vals, line_y, 'r--', linewidth=2, label='Verbindungslinie')
    axs[0].axvline(x=threshold, color='b', linestyle='-', linewidth=2, label=f'Schwellenwert: {threshold:.2f}')
    axs[0].set_xlabel("Range")
    axs[0].set_ylabel("Häufigkeit")
    axs[0].legend()
    axs[0].set_title("Histogramm (getrimmt) und Triangle Methode")
    axs[1].bar(centers_full, hist_full, width=step_full * 0.9, color='gray', label='Histogramm (unbeschnitten)')
    axs[1].set_xlabel("Range")
    axs[1].set_ylabel("Häufigkeit")
    axs[1].legend()
    axs[1].set_title("Gesamtes unbeschnittenes Histogramm")
    plt.tight_layout()
    plt.show()

    print(f"Ermittelter Schwellenwert: {threshold:.2f}")
    return threshold


# =============================================================================
# 2. Datenverarbeitung (Parameter laden, Indizes berechnen, Range-Matrix)
# =============================================================================
def load_parameters(dataset):
    meta_frame = dataset[0][0]
    FOV_vert = 22.5
    FOV_horiz = 360
    vert_total_grids = int(meta_frame.tower.lidars.UPPER_PLATFORM.info.vertical_scanlines)
    horiz_total_grids = int(meta_frame.tower.lidars.UPPER_PLATFORM.info.horizontal_scanlines)
    azimuth_resltn = FOV_horiz / horiz_total_grids
    elevation_resltn = FOV_vert / vert_total_grids
    beam_altitude_angles = meta_frame.tower.lidars.UPPER_PLATFORM.info.beam_altitude_angles
    return FOV_vert, FOV_horiz, vert_total_grids, horiz_total_grids, azimuth_resltn, elevation_resltn, beam_altitude_angles


def get_horizontal_idx(x, y, FOV_horiz, azimuth_resltn, horiz_total_grids):
    measured_azimuth = np.degrees(np.arctan2(y, x))
    measured_azimuth[measured_azimuth < 0] += FOV_horiz
    horizontal_idx = np.floor(measured_azimuth / azimuth_resltn).astype(int) % horiz_total_grids
    return horizontal_idx


def get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles=None):
    measured_elevation = np.degrees(np.arcsin(z / r))
    measured_elevation[measured_elevation < 0] += FOV_vert
    if beam_altitude_angles is not None:
        beam_angles = np.array(beam_altitude_angles)
        differences = np.abs(beam_angles[:, None] - measured_elevation[None, :])
        vertical_idx = np.argmin(differences, axis=0)
    else:
        vertical_idx = np.floor(measured_elevation / elevation_resltn).astype(int) % vert_total_grids
    return vertical_idx


def process_frame(frame, FOV_vert, FOV_horiz, azimuth_resltn, elevation_resltn,
                  horiz_total_grids, vert_total_grids, beam_altitude_angles=None):
    pcl = frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity'], pcl['ring']), axis=-1)
    else:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity']), axis=-1)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    horizontal_idx = get_horizontal_idx(x, y, FOV_horiz, azimuth_resltn, horiz_total_grids)
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        vertical_idx = points[:, 4].astype(int)
    else:
        vertical_idx = get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles)
    return r, horizontal_idx, vertical_idx, None


def update_range_matrix(range_matrix, r, az_idx, el_idx, frame_num):
    for j in range(len(r)):
        if r[j] == 0:
            continue
        if range_matrix[el_idx[j], az_idx[j], frame_num] == 0:
            range_matrix[el_idx[j], az_idx[j], frame_num] = r[j]
        elif range_matrix[el_idx[j], az_idx[j], frame_num] > r[j]:
            range_matrix[el_idx[j], az_idx[j], frame_num] = r[j]
    return


def train_background_model(range_matrix, vert_total_grids, horiz_total_grids):
    range_thrld_matrix = np.ones((vert_total_grids, horiz_total_grids), dtype=np.float32) * 200
    for i in range(vert_total_grids):
        for j in range(horiz_total_grids):
            distances = range_matrix[i, j, :]
            range_thrld_matrix[i, j] = thresholding(distances)
        if i % 10 == 0:
            print(f"Hintergrundtraining: {i} von {vert_total_grids} Zeilen verarbeitet.")
    print("Hintergrundtraining abgeschlossen.")
    return range_thrld_matrix


# =============================================================================
# 3. Hintergrundsubtraktion & Outlier Removal
# =============================================================================
def background_subtraction(test_frame, range_thrld_matrix, FOV_vert, FOV_horiz,
                           azimuth_resltn, elevation_resltn, horiz_total_grids,
                           vert_total_grids, beam_altitude_angles=None):
    pcl = test_frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity'], pcl['ring']), axis=-1)
        rings = points[:, 4].astype(int)
    else:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity']), axis=-1)
        rings = None
    x_test, y_test, z_test = points[:, 0], points[:, 1], points[:, 2]
    r_test = np.sqrt(x_test ** 2 + y_test ** 2 + z_test ** 2)
    horizontal_idx = get_horizontal_idx(x_test, y_test, FOV_horiz, azimuth_resltn, horiz_total_grids)
    if rings is None:
        vertical_idx = get_vertical_idx(z_test, r_test, FOV_vert, elevation_resltn, vert_total_grids,
                                        beam_altitude_angles)
    else:
        vertical_idx = rings
    background_mask = np.array([r_test[k] >= range_thrld_matrix[vertical_idx[k], horizontal_idx[k]]
                                for k in range(len(r_test))])
    foreground_mask = ~background_mask
    foreground_points = np.stack((x_test, y_test, z_test), axis=-1)[foreground_mask]
    background_points = np.stack((x_test, y_test, z_test), axis=-1)[background_mask]
    if rings is not None:
        fg_rings = rings[foreground_mask]
        bg_rings = rings[background_mask]
    else:
        fg_rings, bg_rings = None, None
    return foreground_points, background_points, fg_rings, bg_rings


def apply_outlier_removal(foreground_points, fg_rings, background_points, nb_points=15, radius=0.8):
    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(foreground_points)
    _, ind = fg_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_mask = np.zeros(len(foreground_points), dtype=bool)
    inlier_mask[ind] = True
    new_foreground_points = foreground_points[inlier_mask]
    new_background_points = np.vstack((background_points, foreground_points[~inlier_mask]))
    if fg_rings is not None:
        new_fg_rings = fg_rings[inlier_mask]
        new_bg_rings = np.concatenate((bg_rings, fg_rings[~inlier_mask]))
    else:
        new_fg_rings, new_bg_rings = None, None
    return new_foreground_points, new_background_points, new_fg_rings, new_bg_rings


# =============================================================================
# 4. Clusterung & Oriented Bounding Box via Minimum Bounding Rectangle
# =============================================================================
def minimum_bounding_rectangle(points):
    """
    Berechnet das Minimum Bounding Rectangle (MBR) für ein (N,2)-Array von 2D-Punkten
    mittels Rotating Calipers.

    Rückgabe:
      rectangle: 4x2 Array der Ecken in Reihenfolge.
      angle: Rotationswinkel (in Radiant) des Rechtecks relativ zur x-Achse.
    """
    if points.shape[0] <= 2:
        return points, 0.0
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    min_area = np.inf
    best_rect = None
    best_angle = 0.0
    n = hull_points.shape[0]
    for i in range(n):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        edge = p2 - p1
        angle = -np.arctan2(edge[1], edge[0])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        rot_points = points.dot(R.T)
        min_x = np.min(rot_points[:, 0])
        max_x = np.max(rot_points[:, 0])
        min_y = np.min(rot_points[:, 1])
        max_y = np.max(rot_points[:, 1])
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_angle = angle
            best_rect = np.array([
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y]
            ])
    R_inv = np.array([[np.cos(-best_angle), -np.sin(-best_angle)],
                      [np.sin(-best_angle), np.cos(-best_angle)]])
    rectangle = best_rect.dot(R_inv.T)
    return rectangle, best_angle


def compute_oriented_bbox_for_cluster(cluster_points):
    """
    Berechnet für die übergebenen Cluster-Punkte (Nx3) eine orientierte Bounding Box.
    Es wird die 2D-Projektion auf x-y verwendet, um mittels MBR (Rotating Calipers)
    den Yaw-Winkel zu bestimmen. Die z-Dimension wird über min/max z ergänzt.

    Rückgabe:
      obb: Ein open3d.geometry.OrientedBoundingBox-Objekt.
    """
    if cluster_points.shape[0] == 0:
        return None
    pts_xy = cluster_points[:, :2]
    rect, angle = minimum_bounding_rectangle(pts_xy)
    center_xy = np.mean(rect, axis=0)
    # Breite und Höhe:
    width = np.linalg.norm(rect[0] - rect[3])
    height = np.linalg.norm(rect[0] - rect[1])
    min_z = np.min(cluster_points[:, 2])
    max_z = np.max(cluster_points[:, 2])
    z_extent = max_z - min_z
    center_z = (min_z + max_z) / 2.0
    center = np.array([center_xy[0], center_xy[1], center_z])
    extents = np.array([width, height, z_extent])
    angle = -angle
    R_3d = np.eye(3)
    R_3d[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
    obb = o3d.geometry.OrientedBoundingBox(center, R_3d, extents)
    return obb


def cluster_and_get_oriented_bboxes(foreground_points, eps=0.8, min_points=3):
    """
    Teilt die Foreground-Punktwolke mittels DBSCAN in Cluster auf und berechnet
    für jeden Cluster mittels MBR und PCA eine orientierte Bounding Box.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(foreground_points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"DBSCAN: {max_label + 1} Cluster gefunden (Noise = -1).")
    bboxes = []
    for label in range(max_label + 1):
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            continue
        cluster_points = foreground_points[indices]
        obb = compute_oriented_bbox_for_cluster(cluster_points)
        if obb is not None:
            bboxes.append(obb)
    return bboxes


# =============================================================================
# 5. Visualisierung (Interaktiv & Statisch)
# =============================================================================
def visualize_point_clouds_interactive(foreground_points, background_points,
                                       fg_rings, bg_rings,
                                       FOV_horiz, FOV_vert, azimuth_resltn,
                                       elevation_resltn, horiz_total_grids, vert_total_grids,
                                       range_thrld_matrix, beam_altitude_angles=None,
                                       cluster_bboxes=None):
    """
    Interaktive Visualisierung der kombinierten Punktwolke (Foreground + Background)
    mit optionalen Cluster-Bounding-Boxen (anklickbar).
    """
    fg_len = foreground_points.shape[0]
    bg_len = background_points.shape[0]
    combined_points = np.vstack((foreground_points, background_points)).astype(np.float64)
    colors_fg = np.tile(np.array([[1, 0, 0]], dtype=np.float64), (fg_len, 1))
    colors_bg = np.tile(np.array([[0, 0, 1]], dtype=np.float64), (bg_len, 1))
    combined_colors = np.vstack((colors_fg, colors_bg))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Interaktive Punktwahl (mit BBoxen)")
    vis.add_geometry(pcd, reset_bounding_box=True)
    if cluster_bboxes is not None:
        for bbox in cluster_bboxes:
            bbox.color = (0, 1, 0)
            vis.add_geometry(bbox, reset_bounding_box=True)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()
    if picked_indices is None or len(picked_indices) == 0:
        print("Keine Punkte ausgewählt.")
        return
    combined_np = np.asarray(pcd.points)
    for idx in picked_indices:
        x, y, z = combined_np[idx]
        r_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        hor_idx = get_horizontal_idx(np.array([x]), np.array([y]), FOV_horiz, azimuth_resltn, horiz_total_grids)[0]
        if fg_rings is not None and bg_rings is not None:
            combined_rings = np.concatenate((fg_rings, bg_rings))
            ver_idx = combined_rings[idx]
        else:
            ver_idx = get_vertical_idx(np.array([z]), np.array([r_val]), FOV_vert, elevation_resltn, vert_total_grids,
                                       beam_altitude_angles)[0]
        threshold_val = range_thrld_matrix[ver_idx, hor_idx]
        group = "Foreground" if idx < fg_len else "Background"
        print(
            f"Punkt {idx} ({group}): Range = {r_val:.2f}, hor_idx = {hor_idx}, ver_idx = {ver_idx}, Threshold = {threshold_val:.2f}")


def visualize_bboxes_static_foreground(foreground_points, cluster_bboxes):
    # Erstelle eine Punktwolke aus den Foreground-Punkten und färbe sie rot
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(foreground_points)
    pcd.paint_uniform_color([1, 0, 0])

    # Setze die Farbe der Bounding Boxen (z.B. grün)
    for bbox in cluster_bboxes:
        bbox.color = (0, 1, 0)

    # Visualisiere die Punktwolke zusammen mit den Bounding Boxen
    o3d.visualization.draw_geometries([pcd] + cluster_bboxes,
                                      window_name="Statische Visualisierung der BBoxen (Foreground)")


# =============================================================================
# 6. Hauptprogramm (Integration der Pipeline)
# =============================================================================
def main():
    load_existing_range_model = True
    load_existing_range_matrix = True
    range_model_filename = "range_thrld_matrix.npy"
    range_matrix_filename = "range_matrix.npy"
    dataset_path = "/mnt/hot_data/dataset/coopscenes_anonymized/seq_3"
    total_frame = 4000

    dataset = Dataloader(dataset_path)
    (FOV_vert, FOV_horiz, vert_total_grids, horiz_total_grids,
     azimuth_resltn, elevation_resltn, beam_altitude_angles) = load_parameters(dataset)

    # Range-Matrix laden oder berechnen
    if load_existing_range_matrix and os.path.exists(range_matrix_filename):
        print("Lade vorhandene range_matrix ...")
        range_matrix = np.load(range_matrix_filename)
    else:
        range_matrix = np.zeros((vert_total_grids, horiz_total_grids, total_frame), dtype=np.float32)
        frame_num = 0
        for record in dataset:
            for frame in record:
                if frame_num >= total_frame:
                    break
                r, az_idx, el_idx, _ = process_frame(
                    frame, FOV_vert, FOV_horiz, azimuth_resltn,
                    elevation_resltn, horiz_total_grids, vert_total_grids,
                    beam_altitude_angles)
                update_range_matrix(range_matrix, r, az_idx, el_idx, frame_num)
                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Verarbeitung: {frame_num} Frames verarbeitet.")
            if frame_num >= total_frame:
                break
        print(f"Berechnung der range_matrix abgeschlossen. Gesamtanzahl Frames: {frame_num}")
        np.save(range_matrix_filename, range_matrix)

    # Range-Hintergrundmodell laden oder trainieren
    if load_existing_range_model and os.path.exists(range_model_filename):
        print("Lade vorhandenes Range-Hintergrundmodell ...")
        range_thrld_matrix = np.load(range_model_filename)
    else:
        range_thrld_matrix = train_background_model(range_matrix, vert_total_grids, horiz_total_grids)
        print("Speichere Range-Hintergrundmodell ...")
        np.save(range_model_filename, range_thrld_matrix)

    test_frame = dataset[-1][0]
    foreground_points, background_points, fg_rings, bg_rings = background_subtraction(
        test_frame, range_thrld_matrix, FOV_vert, FOV_horiz,
        azimuth_resltn, elevation_resltn, horiz_total_grids, vert_total_grids,
        beam_altitude_angles)

    print("Führe Outlier Removal (Radius-basiert) durch ...")
    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(foreground_points)
    _, ind = fg_pcd.remove_radius_outlier(nb_points=15, radius=0.8)
    inlier_mask = np.zeros(len(foreground_points), dtype=bool)
    inlier_mask[ind] = True
    new_foreground_points = foreground_points[inlier_mask]
    new_background_points = np.vstack((background_points, foreground_points[~inlier_mask]))
    if fg_rings is not None:
        new_fg_rings = fg_rings[inlier_mask]
        new_bg_rings = np.concatenate((bg_rings, fg_rings[~inlier_mask]))
    else:
        new_fg_rings, new_bg_rings = None, None
    foreground_points = new_foreground_points
    background_points = new_background_points
    fg_rings = new_fg_rings
    bg_rings = new_bg_rings

    print("Führe DBSCAN-Clusterung durch und berechne orientierte Bounding-Boxen ...")
    cluster_bboxes = cluster_and_get_oriented_bboxes(foreground_points, eps=0.8, min_points=3)

    # Interaktive Visualisierung (mit anklickbaren Punkten)
    visualize_point_clouds_interactive(foreground_points, background_points,
                                       fg_rings, bg_rings,
                                       FOV_horiz, FOV_vert,
                                       azimuth_resltn, elevation_resltn,
                                       horiz_total_grids, vert_total_grids,
                                       range_thrld_matrix, beam_altitude_angles,
                                       cluster_bboxes=cluster_bboxes)

    # Zusätzliche statische Visualisierung nur mit den Foreground-Punkten
    visualize_bboxes_static_foreground(foreground_points, cluster_bboxes)


if __name__ == "__main__":
    main()
