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
    Projects each 3D bounding box onto the 2D image plane of the given camera
    and draws the box edges onto the image using PIL.

    Args:
        camera: The camera object (e.g., test_frame.tower.cameras.VIEW_1), with:
            - camera.info.camera_mtx     (3×3 numpy array)
            - camera.info.distortion_mtx (ignored here)
            - get_transformation(camera.info) returns an object whose
              .invert_transformation() has .rotation (3×3) and .translation (3,)
            - camera.image.image: PIL.Image
        bboxes (list[open3d.geometry.OrientedBoundingBox]):
            List of OBBs in the LIDAR coordinate frame.
        line_color (tuple[int,int,int]): RGB color for box edges (default: green).
        line_width (int): Thickness of the lines to draw.

    Returns:
        PIL.Image: The image with projected boxes drawn.
    """
    info = camera.info
    # Intrinsic camera matrix
    cam_mtx = info.camera_mtx.copy()
    # Build extrinsics: LIDAR → camera
    cam_to_lidar = get_transformation(info)
    lidar_to_cam = cam_to_lidar.invert_transformation()
    R = np.array(lidar_to_cam.rotation)  # 3×3
    t = np.array(lidar_to_cam.translation)  # (3,)

    # Get PIL image
    img_pil = camera.image.image.convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    for obb in bboxes:
        # 8 corners in LIDAR frame
        corners = np.asarray(obb.get_box_points())  # shape (8, 3)

        # Transform to camera frame: X_cam = R @ X_lidar + t
        corners_cam = (R @ corners.T + t.reshape(3, 1)).T  # (8, 3)

        # Skip box if all corners are behind camera
        if np.all(corners_cam[:, 2] <= 1e-3):
            continue

        # Project to 2D: u = fx * X/Z + cx, v = fy * Y/Z + cy
        pts_2d_homo = (cam_mtx @ (corners_cam.T / corners_cam[:, 2])).T  # (8, 3)
        pts_2d = pts_2d_homo[:, :2]  # discard scale

        # Round to integer pixel coordinates
        pts_2d_int = np.round(pts_2d).astype(int)

        # Define the 12 edges (corner indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
        ]

        for i, j in edges:
            x1, y1 = pts_2d_int[i]
            x2, y2 = pts_2d_int[j]
            # Only draw if both endpoints are within image bounds
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)

    return img_pil
