#!/usr/bin/env python3
"""
Main entry point for the LIDAR background subtraction pipeline.

Ties together data loading, background modeling, clustering, and visualization,
nun mit Protobuf‐basiertem BackgroundModel für Persistenz.
"""

import os
from configmanager import ConfigManager

from range_intensity import (
    build_range_matrix,
    create_dataset,
)
from background_model import train_background_model, background_subtraction, apply_outlier_removal, BackgroundModel
from clustering_visualization import (
    cluster_and_get_oriented_bboxes,
    project_bboxes_to_image,
    visualize_bboxes_static_foreground,
)


def main():
    """
    Run the entire pipeline:
      1. Load or build range + threshold matrices (BackgroundModel).
      2. Subtract background on test frame.
      3. (Optional) Remove outliers.
      4. Cluster foreground points und compute OBBs.
      5. Visualize results.
    """
    cfg = ConfigManager()
    dataset = create_dataset(cfg.data_format, cfg.data_path)

    # Hintergrundmodell vorbereiten
    bg_model = BackgroundModel(meta=dataset.meta)

    # ---------------------------------------------------------------------
    # 1. Load or build BackgroundModel (enthält SensorMeta + range_matrix + threshold_matrix)
    # ---------------------------------------------------------------------
    if cfg.load_background_model and os.path.exists(cfg.model_path):
        print(f"Loading existing BackgroundModel from {cfg.model_path} ...")
        bg_model.load(cfg.model_path)
    else:
        print("Building range matrix from scratch...")
        bg_model.range_matrix = build_range_matrix(dataset, cfg)

        print("Training background model (Threshold-Matrix)...")
        bg_model.threshold_matrix = train_background_model(
            bg_model.range_matrix,
            bg_model.meta,
        )

        print(f"Saving BackgroundModel to {cfg.model_path} ...")
        bg_model.save(cfg.model_path)
    print('test')

    # ---------------------------------------------------------------------
    # 2. Background subtraction on the last frame
    # ---------------------------------------------------------------------
    test_frame = dataset[-1]
    fg_pts, bg_pts, fg_rings, bg_rings = background_subtraction(
        test_frame,
        bg_model
    )

    # ---------------------------------------------------------------------
    # 3. Optional: Outlier Removal
    # ---------------------------------------------------------------------
    print("Applying radius‐based Outlier Removal...")
    fg_pts, bg_pts, fg_rings, bg_rings = apply_outlier_removal(
        fg_pts,
        fg_rings,
        bg_pts,
        bg_rings,
        nb_points=5,
        radius=0.7,
    )

    # ---------------------------------------------------------------------
    # 4. Cluster foreground points und orientierte Bounding‐Boxen berechnen
    # ---------------------------------------------------------------------
    print("Clustering and computing oriented bounding boxes...")
    cluster_bboxes = cluster_and_get_oriented_bboxes(fg_pts, eps=2.5, min_points=5)

    # ---------------------------------------------------------------------
    # 5. Projektion der Bounding‐Boxen auf die VIEW_1‐Kamera
    # ---------------------------------------------------------------------
    print("Projecting 3D bounding boxes onto the image...")
    view1 = test_frame.tower.cameras.VIEW_1
    projected_pil = project_bboxes_to_image(view1, cluster_bboxes)

    # POPUP‐Fenster mit PIL‐Default‐Viewer
    projected_pil.show(title="Projected Bounding Boxes")

    # (Optional) Statische Anzeige nur der Bounding‐Boxen auf den FG‐Punkten:
    # visualize_bboxes_static_foreground(fg_pts, cluster_bboxes)


if __name__ == "__main__":
    main()
