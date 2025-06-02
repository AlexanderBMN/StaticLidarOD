#!/usr/bin/env python3
"""
Main entry point for the LIDAR background subtraction pipeline.

Ties together data loading, background modeling, clustering, and visualization.
"""

import os

import numpy as np
from dataloader import build_range_matrix, load_parameters
from background_model import train_background_model, background_subtraction, apply_outlier_removal
from clustering_visualization import (
    cluster_and_get_oriented_bboxes,
    visualize_point_clouds_interactive,
    project_bboxes_to_image,
    visualize_bboxes_static_foreground,
)
from coopscenes import Dataloader


def main():
    """
    Run the entire pipeline:
      1. Load or build range matrix.
      2. Load or train background model.
      3. Subtract background on test frame.
      4. (Optional) Remove outliers.
      5. Cluster foreground points and compute OBBs.
      6. Visualize results.
    """
    load_existing_range_model = True
    load_existing_range_matrix = True

    range_model_filename = r"C:\Users\abaum\PycharmProjects\PythonProject\StaticLidarOD\data\range_thrld_matrix.npy"
    range_matrix_filename = r"C:\Users\abaum\PycharmProjects\PythonProject\StaticLidarOD\data\range_matrix.npy"
    dataset_path = r"C:\Users\abaum\PycharmProjects\PythonProject\StaticLidarOD\data"
    total_frame = 4000

    dataset = Dataloader(dataset_path)

    # 1. Load or build range matrix
    if load_existing_range_matrix and os.path.exists(range_matrix_filename):
        print("Loading existing range matrix...")
        range_matrix = np.load(range_matrix_filename)
        # We still need sensor parameters for the test frame
        params = load_parameters(dataset)
        (FOV_vert, FOV_horiz, vert_total_grids,
         horiz_total_grids, azimuth_resltn,
         elevation_resltn, beam_altitude_angles) = params
    else:
        print("Building range matrix...")
        range_matrix, params = build_range_matrix(dataset, total_frame)
        (FOV_vert, FOV_horiz, vert_total_grids,
         horiz_total_grids, azimuth_resltn,
         elevation_resltn, beam_altitude_angles) = params
        print("Saving computed range matrix...")
        np.save(range_matrix_filename, range_matrix)

    # 2. Load or train background model
    if load_existing_range_model and os.path.exists(range_model_filename):
        print("Loading existing background model...")
        range_thrld_matrix = np.load(range_model_filename)
    else:
        print("Training background model...")
        range_thrld_matrix = train_background_model(range_matrix, vert_total_grids, horiz_total_grids)
        print("Saving background model...")
        np.save(range_model_filename, range_thrld_matrix)

    # 3. Background subtraction on the last frame
    test_frame = dataset[-1][0]
    foreground_points, background_points, fg_rings, bg_rings = background_subtraction(
        test_frame,
        range_thrld_matrix,
        FOV_vert,
        FOV_horiz,
        azimuth_resltn,
        elevation_resltn,
        horiz_total_grids,
        vert_total_grids,
        beam_altitude_angles,
    )

    # 4. Apply outlier removal (optional)
    print("Führe Outlier Removal (Radius-basiert) durch ...")
    # Wenn Du Outlier Removal aktivieren möchtest, hebe den Kommentar auf:
    foreground_points, background_points, fg_rings, bg_rings = apply_outlier_removal(
        foreground_points,
        fg_rings,
        background_points,
        bg_rings,
        nb_points=5,
        radius=0.7
    )

    # 5. Cluster foreground points and compute oriented bounding boxes
    print("Clustering and computing oriented bounding boxes...")
    cluster_bboxes = cluster_and_get_oriented_bboxes(foreground_points, eps=2.5, min_points=5)
    '''
    # 6. Interactive visualization
    print("Launching interactive visualization...")
    visualize_point_clouds_interactive(
        foreground_points,
        background_points,
        fg_rings,
        bg_rings,
        FOV_horiz,
        FOV_vert,
        azimuth_resltn,
        elevation_resltn,
        horiz_total_grids,
        vert_total_grids,
        range_matrix,
        range_thrld_matrix,
        beam_altitude_angles,
        cluster_bboxes=cluster_bboxes,
    )

    # To visualize static bounding boxes only:
    visualize_bboxes_static_foreground(foreground_points, cluster_bboxes)
    '''
    # 6. Project OBBs onto VIEW_1 image using PIL
    print("Projecting 3D bounding boxes onto the image...")
    view1 = test_frame.tower.cameras.VIEW_1
    projected_pil = project_bboxes_to_image(view1, cluster_bboxes)

    # Display projected result (uses PIL's default viewer)
    projected_pil.show(title="Projected Bounding Boxes")


if __name__ == "__main__":
    main()
