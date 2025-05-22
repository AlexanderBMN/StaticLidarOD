import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import open3d as o3d


def normalize_angle(angle):
    """Normalisiert einen Winkel (in Radiant) in den Bereich [-pi/2, pi/2]."""
    while angle < -np.pi / 2:
        angle += np.pi
    while angle > np.pi / 2:
        angle -= np.pi
    return angle


def minimum_bounding_rectangle(points):
    """
    Berechnet das Minimum Bounding Rectangle (MBR) für ein (N,2)-Array von 2D-Punkten
    mittels Rotating Calipers.

    Rückgabe:
      rectangle: 4x2 Array der Ecken in Originalkoordinaten.
      angle: Normalisierter Rotationswinkel (in Radiant) relativ zur x-Achse.
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
        angle = normalize_angle(angle)
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
            # Im rotierten System wäre das Rechteck:
            best_rect = np.array([
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y]
            ])

    # Rücktransformation der Rechtecksecken in das Originalsystem
    R_inv = np.array([[np.cos(-best_angle), -np.sin(-best_angle)],
                      [np.sin(-best_angle), np.cos(-best_angle)]])
    rectangle = best_rect.dot(R_inv.T)

    return rectangle, best_angle


def compute_oriented_bbox_for_cluster(cluster_points):
    """
    Berechnet für die übergebenen Cluster-Punkte (Nx3) eine orientierte 3D-Bounding Box.
    Es wird zuerst die 2D-Projection (x-y) betrachtet, um mittels MBR den Yaw-Winkel zu
    bestimmen. Dann werden die Ausdehnungen in x-y über das rotierten System bestimmt,
    und die z-Dimension über min/max z ergänzt.

    Rückgabe:
      obb: Ein open3d.geometry.OrientedBoundingBox-Objekt.
    """
    if cluster_points.shape[0] == 0:
        return None

    pts_xy = cluster_points[:, :2]
    # Bestimme MBR und Winkel in 2D
    rect, angle = minimum_bounding_rectangle(pts_xy)

    # Erzeuge Rotationsmatrix R aus dem berechneten Winkel
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    # Transformiere die 2D-Punkte in das rotierte System
    rot_pts = pts_xy.dot(R.T)

    # Bestimme im rotierten System die Grenzen und somit die Ausdehnung
    min_x, max_x = np.min(rot_pts[:, 0]), np.max(rot_pts[:, 0])
    min_y, max_y = np.min(rot_pts[:, 1]), np.max(rot_pts[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Berechne das Zentrum im rotierten System
    center_rot = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    # Transformiere das Zentrum zurück in das Originalsystem
    center_xy = center_rot.dot(R)

    # Bestimme die z-Ausdehnung
    min_z = np.min(cluster_points[:, 2])
    max_z = np.max(cluster_points[:, 2])
    z_extent = max_z - min_z
    center_z = (min_z + max_z) / 2.0

    center = np.array([center_xy[0], center_xy[1], center_z])
    extents = np.array([width, height, z_extent])

    # Erstelle eine 3D-Rotationsmatrix, wobei nur in der x-y-Ebene rotiert wird
    R_3d = np.eye(3)
    R_3d[:2, :2] = R

    obb = o3d.geometry.OrientedBoundingBox(center, R_3d, extents)
    return obb


def synthetic_test_3d():
    """
    Erzeugt einen synthetischen 3D-Cluster (rechteckige Punktwolke in x-y mit z-Variationen),
    rotiert diesen in der x-y-Ebene mit einem bekannten Winkel und berechnet die 3D-OBB.
    Anschließend wird die Punktwolke zusammen mit der OBB in einer Open3D-Szene visualisiert.
    """
    # Bekannter Rotationswinkel (z.B. 30°)
    true_angle = np.deg2rad(30)
    width, height, depth = 5, 3, 2  # Dimensionen des Clusters

    # Erzeuge ein Gitter in der x-y-Ebene
    xs = np.linspace(-width / 2, width / 2, num=20)
    ys = np.linspace(-height / 2, height / 2, num=20)
    xv, yv = np.meshgrid(xs, ys)
    points_2d = np.column_stack([xv.ravel(), yv.ravel()])

    # Erzeuge zufällige z-Werte (um etwas Variation in z zu simulieren)
    zv = np.random.uniform(-depth / 2, depth / 2, size=points_2d.shape[0])
    points_3d = np.column_stack([points_2d, zv])

    # Rotiere die Punkte in der x-y-Ebene
    R_true = np.array([[np.cos(true_angle), -np.sin(true_angle)],
                       [np.sin(true_angle), np.cos(true_angle)]])
    rotated_xy = points_2d.dot(R_true.T)
    rotated_points_3d = np.column_stack([rotated_xy, zv])

    # Erzeuge und färbe die Punktwolke
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_points_3d)
    pcd.paint_uniform_color([0.1, 0.1, 0.7])

    # Berechne die 3D-orientierte Bounding Box
    obb = compute_oriented_bbox_for_cluster(rotated_points_3d)
    if obb is None:
        print("Keine OBB berechnet.")
        return
    obb.color = (0, 1, 0)  # Farbe der Box (grün)

    # Visualisiere in Open3D
    o3d.visualization.draw_geometries([pcd, obb],
                                      window_name="3D Visualisierung: Punktwolke und OBB")

    print("Wahrer Winkel (deg): {:.2f}".format(np.degrees(true_angle)))
    # Wir können auch den berechneten Winkel aus der 2D-MBR-Funktion ausgeben:
    rect, computed_angle = minimum_bounding_rectangle(rotated_points_3d[:, :2])
    print("Berechneter Winkel (deg): {:.2f}".format(np.degrees(computed_angle)))


# Führe den synthetischen 3D-Test aus
synthetic_test_3d()
