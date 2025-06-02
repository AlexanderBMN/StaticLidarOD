#!/usr/bin/env python3
"""
Module: clustering_visualization.py

Contains DBSCAN clustering, oriented bounding box computation,
and all visualization routines (coarse/fine histograms, interactive/static views).
"""

import numpy as np
import open3d as o3d
from PIL import ImageDraw
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from thresholding import compute_threshold_meta

from coopscenes import get_transformation


def minimum_bounding_rectangle(points_2d: np.ndarray):
    """
    Compute the minimum bounding rectangle (MBR) for 2D points via rotating calipers.
    """
    if points_2d.shape[0] <= 2:
        return points_2d, 0.0

    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    min_area = np.inf
    best_rect = None
    best_angle = 0.0
    n = hull_points.shape[0]

    for i in range(n):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        edge = p2 - p1
        angle = -np.arctan2(edge[1], edge[0])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rot_points = points_2d.dot(R.T)
        min_x, max_x = np.min(rot_points[:, 0]), np.max(rot_points[:, 0])
        min_y, max_y = np.min(rot_points[:, 1]), np.max(rot_points[:, 1])
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_angle = angle
            best_rect = np.array([[max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]])

    R_inv = np.array([[np.cos(-best_angle), -np.sin(-best_angle)], [np.sin(-best_angle), np.cos(-best_angle)]])
    rectangle = best_rect.dot(R_inv.T)
    return rectangle, best_angle


def compute_oriented_bbox_for_cluster(cluster_points: np.ndarray):
    """
    Compute an oriented 3D bounding box for a cluster of 3D points.
    """
    if cluster_points.shape[0] == 0:
        return None

    pts_xy = cluster_points[:, :2]
    rect, angle = minimum_bounding_rectangle(pts_xy)
    center_xy = np.mean(rect, axis=0)

    width = np.linalg.norm(rect[0] - rect[3])
    height = np.linalg.norm(rect[0] - rect[1])
    min_z, max_z = np.min(cluster_points[:, 2]), np.max(cluster_points[:, 2])
    z_extent = max_z - min_z
    center_z = (min_z + max_z) / 2.0

    center = np.array([center_xy[0], center_xy[1], center_z])
    extents = np.array([width, height, z_extent])
    angle = -angle

    R_3d = np.eye(3)
    R_3d[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    obb = o3d.geometry.OrientedBoundingBox(center, R_3d, extents)
    return obb


def cluster_and_get_oriented_bboxes(foreground_points: np.ndarray, eps: float = 0.8, min_points: int = 3):
    """
    Cluster foreground points with DBSCAN and compute oriented 3D bounding boxes.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(foreground_points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"DBSCAN: Found {max_label + 1} clusters (noise = -1).")

    bboxes = []
    for label in range(max_label + 1):
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            continue
        cluster_pts = foreground_points[indices]
        obb = compute_oriented_bbox_for_cluster(cluster_pts)
        if obb is not None:
            bboxes.append(obb)
    return bboxes


def visualize_coarse_histogram(distances: np.ndarray):
    """
    Plot the coarse histogram with trim annotation using compute_threshold_meta.
    """
    meta = compute_threshold_meta(distances)
    if 'coarse' not in meta:
        print("No valid distances for coarse visualization.")
        return

    hist2, edges2, sel_coarse, trim_thr = (
        meta['coarse']['hist'],
        meta['coarse']['edges'],
        meta['coarse']['sel'],
        meta['coarse']['trim_thr'],
    )
    centers_coarse = (edges2[:-1] + edges2[1:]) / 2
    width = edges2[1] - edges2[0]

    plt.figure(figsize=(8, 4))
    plt.bar(centers_coarse, hist2, width=width * 0.9, color='lightgray', edgecolor='k', label='Coarse Histogram')
    plt.axvline(trim_thr, color='b', linestyle='--', label=f'Coarse Trim = {trim_thr:.2f} m')
    plt.scatter([centers_coarse[sel_coarse]], [hist2[sel_coarse]], color='k', zorder=5,
                label=f'Peak @{centers_coarse[sel_coarse]:.2f} m ({hist2[sel_coarse]} pts)')
    plt.xlabel('Range [m]')
    plt.ylabel('Count')
    plt.title('Coarse Histogram with Trim (Top-3 ≥ 20%)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_cfta(distances: np.ndarray):
    """
    Plot the fine-resolution histogram (CFTA) with triangle line and threshold.
    """
    meta = compute_threshold_meta(distances)
    if 'fine' not in meta:
        print(f"Threshold = {meta.get('threshold'):.2f} m")
        return

    adjusted_hist = meta['fine']['hist']
    edges_fine = meta['fine']['edges']
    centers = meta['fine']['centers']
    sel_fine = meta['fine']['sel']
    thr = meta['threshold']

    x0, y0 = centers[0], adjusted_hist[0]
    x1, y1 = centers[sel_fine], adjusted_hist[sel_fine]

    plt.figure(figsize=(8, 4))
    plt.bar(centers, adjusted_hist, width=(edges_fine[1] - edges_fine[0]) * 0.9,
            color='lightgray', label='Fine Histogram')
    plt.plot([x0, x1], [y0, y1], 'r--', label='Triangle Line')
    plt.axvline(thr, linestyle='-', label=f'Threshold = {thr:.2f} m')
    plt.scatter([x1], [y1], color='k', zorder=5,
                label=f'Selected Peak @{x1:.2f} m ({int(adjusted_hist[sel_fine])} pts)')
    plt.xlabel('Range [m]')
    plt.ylabel('Frequency')
    plt.title('CFTA Fine Histogram (Top-5 ≥ 10%)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_point_clouds_interactive(foreground_points: np.ndarray, background_points: np.ndarray,
                                       fg_rings, bg_rings,
                                       FOV_horiz, FOV_vert, azimuth_resltn,
                                       elevation_resltn, horiz_total_grids, vert_total_grids,
                                       range_matrix, range_thrld_matrix,
                                       beam_altitude_angles=None,
                                       cluster_bboxes=None):
    """
    Interactive visualization of combined point cloud with cluster bounding boxes.
    Clicking a point triggers coarse & fine histogram plots for that beam.
    """
    fg_len = foreground_points.shape[0]
    bg_len = background_points.shape[0]
    combined_points = np.vstack((foreground_points, background_points)).astype(np.float64)
    colors_fg = np.tile([1, 0, 0], (fg_len, 1))
    colors_bg = np.tile([0, 0, 1], (bg_len, 1))
    combined_colors = np.vstack((colors_fg, colors_bg))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Interactive Point Selection (CFTA)")
    vis.add_geometry(pcd, reset_bounding_box=True)
    if cluster_bboxes:
        for bbox in cluster_bboxes:
            bbox.color = (0, 1, 0)
            vis.add_geometry(bbox, reset_bounding_box=True)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if not picked_indices:
        print("No points selected.")
        return

    from dataloader import get_horizontal_idx, get_vertical_idx

    for idx in picked_indices:
        x, y, z = combined_points[idx]
        r_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        hor_idx = get_horizontal_idx(np.array([x]), np.array([y]), FOV_horiz,
                                     azimuth_resltn, horiz_total_grids)[0]
        if fg_rings is not None and bg_rings is not None:
            all_rings = np.concatenate((fg_rings, bg_rings))
            ver_idx = all_rings[idx]
        else:
            ver_idx = get_vertical_idx(np.array([z]), np.array([r_val]), FOV_vert,
                                       elevation_resltn, vert_total_grids,
                                       beam_altitude_angles)[0]

        thr_val = range_thrld_matrix[ver_idx, hor_idx]
        group = "Foreground" if idx < fg_len else "Background"
        print(f"Point {idx} ({group}): Range={r_val:.2f} m, hor_idx={hor_idx}, "
              f"ver_idx={ver_idx}, Threshold={thr_val:.2f} m")

        distances = range_matrix[ver_idx, hor_idx, :]

        print(f"Plotting coarse histogram for beam (ver={ver_idx}, hor={hor_idx})...")
        visualize_coarse_histogram(distances)

        print(f"Plotting CFTA fine histogram for beam (ver={ver_idx}, hor={hor_idx})...")
        visualize_cfta(distances)


def visualize_bboxes_static_foreground(foreground_points: np.ndarray, cluster_bboxes):
    """
    Static visualization of foreground points with cluster bounding boxes.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(foreground_points)
    pcd.paint_uniform_color([1, 0, 0])

    for bbox in cluster_bboxes:
        bbox.color = (0, 1, 0)

    o3d.visualization.draw_geometries(
        [pcd] + cluster_bboxes, window_name="Static Foreground BBoxes"
    )


def project_bboxes_to_image(camera, bboxes, line_color=(0, 255, 0), line_width=2):
    """
    Projiziert und zeichnet nur die 3D-Bounding-Boxes, die tatsächlich teilweise im Bild liegen.
    Sortiert dabei die 8 Eckpunkte so, dass 0–3 = untere Ebene (z_min), 4–7 = obere Ebene (z_max),
    und verbindet sie in der Standard-Reihenfolge.

    Args:
        camera: test_frame.tower.cameras.VIEW_1 (mit .info und .image.image)
            • .info.camera_mtx     = 3×3 Intrinsik
            • .image.image         = PIL.Image
            • get_transformation(info).invert_transformation().mtx liefert 4×4 LIDAR→Camera.
        bboxes: Liste von open3d.geometry.OrientedBoundingBox (im LIDAR-Koordinatenrahmen).
        line_color: RGB‐Triple für die Linien (default grün).
        line_width: Liniendicke (default 2px).

    Returns:
        PIL.Image: Das Bild, in das nur die tatsächlich sichtbaren Boxkanten eingezeichnet wurden.
    """
    info = camera.info
    cam_mtx = info.camera_mtx.copy()

    # Extrinsik (LIDAR→Kamera)
    lidar_to_cam_tf = get_transformation(info).invert_transformation()
    T = lidar_to_cam_tf.mtx
    R_mat = T[:3, :3].copy()
    t_vec = T[:3, 3].copy()

    # PIL-Bild zum Zeichnen vorbereiten
    img_pil = camera.image.image.convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    width, height = img_pil.size

    for idx, obb in enumerate(bboxes):
        # 1) Lade die 8 Eckpunkte (LIDAR-Rahmen)
        corners = np.asarray(obb.get_box_points())  # shape (8,3)

        # 2) Untere vs. obere Ebene nach z‐Werten trennen
        z_vals = corners[:, 2]
        z_min = z_vals.min()
        z_max = z_vals.max()

        idx_bottom = np.where(np.isclose(z_vals, z_min, atol=1e-6))[0]
        idx_top = np.where(np.isclose(z_vals, z_max, atol=1e-6))[0]

        # Falls weniger bzw. mehr als 4 Punkte je Ebene, skip
        if len(idx_bottom) != 4 or len(idx_top) != 4:
            continue

        # Hilfsfunktion: Sortiere vier Eckpunkte "gegen den Uhrzeigersinn" in xy-Ebene
        def sort_ccw(index_list):
            pts_xy = corners[index_list, :2]  # (4,2)
            center_xy = pts_xy.mean(axis=0)  # (2,)
            angles = np.arctan2(pts_xy[:, 1] - center_xy[1],
                                pts_xy[:, 0] - center_xy[0])
            sorted_idx_inner = np.argsort(angles)  # 4‐Elemente‐Permutation
            return index_list[sorted_idx_inner]  # (4,)

        sorted_bottom = sort_ccw(idx_bottom)  # Reihenfolge der unteren 4 Ecken
        sorted_top = sort_ccw(idx_top)  # Reihenfolge der oberen 4 Ecken

        # 3) Erstelle nun eine durchgehende Reihenfolge [b0,b1,b2,b3, t0,t1,t2,t3]
        ordered_indices = np.hstack((sorted_bottom, sorted_top))  # (8,)

        # 4) Extrahiere in dieser Reihenfolge die Ecken
        corners_ordered = corners[ordered_indices]  # (8,3)

        # 5) Transformation in Kamera‐Koordinaten
        corners_cam = (R_mat @ corners_ordered.T).T + t_vec  # (8,3)

        # 6) Prüfen, ob überhaupt eine Ecke vor der Kamera liegt
        z_cam_vals = corners_cam[:, 2]
        if np.all(z_cam_vals <= 1e-6):
            # Alle Ecken hinter Kamera → skip
            continue

        # 7) Projiziere in Bildkoordinaten
        Z = z_cam_vals.reshape(-1, 1)  # (8,1)
        pts_norm = corners_cam[:, :2] / Z  # (8,2)
        hom = np.hstack((pts_norm, np.ones((8, 1))))  # (8,3)
        uv = (cam_mtx @ hom.T).T  # (8,3)
        uv2d = uv[:, :2]  # (8,2) floats
        uv_int = np.round(uv2d).astype(int)  # (8,2) ints

        # 8) Prüfen, ob mindestens ein Eckpunkt in den Bildbereich fällt
        in_frame = [
            (0 <= uv_int[i, 0] < width) and (0 <= uv_int[i, 1] < height) and (z_cam_vals[i] > 1e-6)
            for i in range(8)
        ]
        if not any(in_frame):
            # Keine Ecke sichtbar → skip
            continue

        # 10) Kantenpaare (nun, da 0–3 unten, 4–7 oben, stimmen diese Paare)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # untere Fläche
            (4, 5), (5, 6), (6, 7), (7, 4),  # obere Fläche
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertikale Kanten
        ]

        # 11) Zeichne jede Kante nur, falls beide Endpunkte im Bild liegen
        for (i1, i2) in edges:
            u1, v1 = uv_int[i1]
            u2, v2 = uv_int[i2]
            if (0 <= u1 < width and 0 <= v1 < height and
                    0 <= u2 < width and 0 <= v2 < height and
                    z_cam_vals[i1] > 1e-6 and z_cam_vals[i2] > 1e-6):
                draw.line([(u1, v1), (u2, v2)], fill=line_color, width=line_width)

    return img_pil
