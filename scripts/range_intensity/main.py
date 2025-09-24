#!/usr/bin/env python3
"""
Main entry point for the LIDAR background subtraction pipeline.

Ties together data loading, background modeling, clustering, and visualization,
nun mit Protobuf‐basiertem BackgroundModel für Persistenz.
"""

from range_intensity import BackgroundModel
from range_intensity.core import background_subtraction, apply_outlier_removal
from range_intensity.viz.clustering_visualization import (
    cluster_and_get_oriented_bboxes,
    visualize_point_clouds_interactive
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

    # ---------------------------------------------------------------------
    # 1. Load or build BackgroundModel (enthält config + dataset + range_matrix + threshold_matrix)
    # ---------------------------------------------------------------------
    bg_model = BackgroundModel()
    # ---------------------------------------------------------------------
    # 2. Background subtraction on the last frame
    # ---------------------------------------------------------------------
    test_frame = bg_model.dataset.get_test_frame()
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
        bg_model.config.outlier
    )

    # ---------------------------------------------------------------------
    # 4. Cluster foreground points und orientierte Bounding‐Boxen berechnen
    # ---------------------------------------------------------------------
    print("Clustering and computing oriented bounding boxes...")
    cluster_bboxes = cluster_and_get_oriented_bboxes(fg_pts, bg_model.config.clustering)

    # ---------------------------------------------------------------------
    # 5. Projektion der Bounding‐Boxen auf die VIEW_1‐Kamera
    # ---------------------------------------------------------------------
    print("Projecting 3D bounding boxes onto the image...")
    # view1 = test_frame.tower.cameras.VIEW_1
    # projected_pil = project_bboxes_to_image(view1, cluster_bboxes)

    # POPUP‐Fenster mit PIL‐Default‐Viewer
    # projected_pil.show(title="Projected Bounding Boxes")

    # (Optional) Statische Anzeige nur der Bounding‐Boxen auf den FG‐Punkten:
    # visualize_bboxes_static_foreground(fg_pts, cluster_bboxes)

    visualize_point_clouds_interactive(fg_pts, bg_pts, fg_rings, bg_rings, bg_model, cluster_bboxes)


if __name__ == "__main__":
    main()
