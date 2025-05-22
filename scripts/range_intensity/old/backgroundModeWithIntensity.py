#!/usr/bin/env python3
import numpy as np
import os
from coopscenes import Dataloader
import open3d as o3d
import matplotlib.pyplot as plt


# ==========================================
# 1. Hilfsfunktion: thresholding (CFTA)
# ==========================================
def thresholding(distances, dist_max=200, fine_bin_num=200):
    """
    Bestimmt einen Schwellwert (Threshold) für die Hintergrund-/Vordergrundtrennung
    basierend auf einem Coarse-Fine Triangle Algorithm (CFTA).

    Parameter:
    - distances: 1D-Array mit Distanzwerten (z.B. Range-Werte eines LiDAR-Scans)
    - dist_max: maximal zulässiger Wert, falls keine sinnvollen Werte gefunden werden (Default: 200)
    - fine_bin_num: Anzahl feiner Bins für den feinen Histogramm-Teil (Default: 200)

    Rückgabe:
    - threshold: Bestimmter Schwellwert, um Hintergrund von Vordergrund zu trennen.
    """

    # ------------------------------
    # 1. Vorverarbeitung: Null-Werte entfernen
    # ------------------------------
    original_frame_size = len(distances)
    non_zero_distances = distances[distances > 0]  # nur gültige (nicht 0) Werte behalten
    if len(non_zero_distances) == 0:
        return dist_max  # falls keine Werte vorhanden, gib den maximalen Wert zurück
    if len(non_zero_distances) < original_frame_size * 0.2:
        # Wenn weniger als 20 % der ursprünglichen Werte gültig sind, nutze das Maximum
        return np.max(non_zero_distances)
    distances = non_zero_distances

    # ------------------------------
    # 2. Grobes Histogramm (Coarse Step)
    # ------------------------------
    n_bins = max(int(round(np.max(distances))), 1)
    N_coarse, edges_coarse = np.histogram(distances, bins=n_bins)
    max_bin_idx = np.argmax(N_coarse)
    if max_bin_idx != len(edges_coarse) - 2:
        coarse_bin_interval = edges_coarse[1] - edges_coarse[0]
        dist_limit = edges_coarse[max_bin_idx] + coarse_bin_interval * 5
        distances = distances[distances <= dist_limit]
    if len(distances) < 2:
        return dist_max

    # ------------------------------
    # 3. Feines Histogramm (Fine Step)
    # ------------------------------
    max_range = np.max(distances)
    bin_half = round(max_range / fine_bin_num, 2)
    if bin_half == 0:
        bin_half = 0.01
    step = bin_half * 2
    bins = np.arange(-bin_half, max_range + step, step)
    hist, edges = np.histogram(distances, bins=bins)
    hist = hist + 1
    lehisto = hist

    # ------------------------------
    # 4. Triangle-Methode: Bestimme den optimalen Schwellwert
    # ------------------------------
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


# ==========================================
# 2. Funktionen zur Verarbeitung der Daten
# ==========================================
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
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    intensities = points[:, 3]
    horizontal_idx = get_horizontal_idx(x, y, FOV_horiz, azimuth_resltn, horiz_total_grids)
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        vertical_idx = points[:, 4].astype(int)
    else:
        vertical_idx = get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles)
    return r, horizontal_idx, vertical_idx, intensities


def update_range_matrix(range_matrix, r, az_idx, el_idx, frame_num):
    for j in range(len(r)):
        if r[j] == 0:
            continue
        if range_matrix[el_idx[j], az_idx[j], frame_num] == 0:
            range_matrix[el_idx[j], az_idx[j], frame_num] = r[j]
        elif range_matrix[el_idx[j], az_idx[j], frame_num] > r[j]:
            range_matrix[el_idx[j], az_idx[j], frame_num] = r[j]
    return


def update_intensity_matrix(intensity_matrix, intensities, az_idx, el_idx, frame_num):
    for j in range(len(intensities)):
        if intensities[j] == 0:
            continue
        if intensity_matrix[el_idx[j], az_idx[j], frame_num] == 0:
            intensity_matrix[el_idx[j], az_idx[j], frame_num] = intensities[j]
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


def background_subtraction(test_frame, range_thrld_matrix, FOV_vert, FOV_horiz,
                           azimuth_resltn, elevation_resltn, horiz_total_grids,
                           vert_total_grids, beam_altitude_angles=None):
    pcl = test_frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity'], pcl['ring']), axis=-1)
    else:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity']), axis=-1)
    x_test = points[:, 0]
    y_test = points[:, 1]
    z_test = points[:, 2]
    r_test = np.sqrt(x_test ** 2 + y_test ** 2 + z_test ** 2)
    horizontal_idx = get_horizontal_idx(x_test, y_test, FOV_horiz, azimuth_resltn, horiz_total_grids)
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        vertical_idx = points[:, 4].astype(int)
    else:
        vertical_idx = get_vertical_idx(z_test, r_test, FOV_vert, elevation_resltn, vert_total_grids,
                                        beam_altitude_angles)
    background_mask = np.array([r_test[k] >= range_thrld_matrix[vertical_idx[k], horizontal_idx[k]]
                                for k in range(len(r_test))])
    foreground_mask = ~background_mask
    foreground_points = np.stack((x_test, y_test, z_test), axis=-1)[foreground_mask]
    background_points = np.stack((x_test, y_test, z_test), axis=-1)[background_mask]
    return foreground_points, background_points


def visualize_point_clouds_interactive(foreground_points, background_points,
                                       FOV_horiz, FOV_vert, azimuth_resltn,
                                       elevation_resltn, horiz_total_grids, vert_total_grids,
                                       range_thrld_matrix, beam_altitude_angles=None):
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
    vis.create_window(window_name="Interaktive Punktwahl")
    vis.add_geometry(pcd, reset_bounding_box=True)
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
        if beam_altitude_angles is not None:
            ver_idx = get_vertical_idx(np.array([z]), np.array([r_val]), FOV_vert, elevation_resltn, vert_total_grids,
                                       beam_altitude_angles)[0]
        else:
            ver_idx = int(np.floor(np.degrees(np.arcsin(z / r_val)) / elevation_resltn)) % vert_total_grids
        threshold_val = range_thrld_matrix[ver_idx, hor_idx]
        group = "Foreground" if idx < fg_len else "Background"
        print(
            f"Punkt {idx} ({group}): Range = {r_val:.2f}, hor_idx = {hor_idx}, ver_idx = {ver_idx}, Threshold = {threshold_val:.2f}")


# ==========================================
# DMD-basierte Intensitäts-Hintergrundsubtraktion (mit Parameter "rank")
# ==========================================
def dmd_background_subtraction(intensity_img, rank=2):
    """
    Führt eine DMD-Zerlegung auf der 2D-Intensity-Matrix (n x m) durch,
    um den statischen Hintergrund (Hintergrundmodus) zu extrahieren.

    Parameter:
    - intensity_img: 2D-Array (n = Anzahl vertikaler Gitter, m = Anzahl Frames)
    - rank: Anzahl der zu verwendenden Modi (Standard: 1)

    Rückgabe:
    - bg_img: Hintergrund-Intensity als 2D-Array (n x m)
    """
    X = intensity_img[:, :-1]
    X_prime = intensity_img[:, 1:]
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Ur = U[:, :rank]
    Sigmar = np.diag(Sigma[:rank])
    Vr = VT[:rank, :]
    A_tilde = Ur.T @ X_prime @ Vr.T @ np.linalg.inv(Sigmar)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = X_prime @ Vr.T @ np.linalg.inv(Sigmar) @ W
    y0, _, _, _ = np.linalg.lstsq(Phi, intensity_img[:, 0], rcond=None)
    m = intensity_img.shape[1]
    time_dynamics = np.zeros((rank, m), dtype=complex)
    omega = np.log(eigvals)
    for i in range(m):
        time_dynamics[:, i] = y0 * np.exp(omega * i)
    v_dmd = Phi @ time_dynamics
    v_dmd = np.abs(v_dmd)
    background_mode = v_dmd[:, 0]  # Wähle den dominanten, statischen Modus
    bg_img = np.tile(background_mode[:, np.newaxis], (1, intensity_img.shape[1]))
    return bg_img


# ==========================================
# Intensitätsbasierte Hintergrundsubtraktion (Segmentierung)
# ==========================================
def background_subtraction_intensity(test_frame, bg_intensity, frame_idx, FOV_vert, FOV_horiz,
                                     azimuth_resltn, elevation_resltn, horiz_total_grids, vert_total_grids,
                                     beam_altitude_angles=None, tol=0.3):
    """
    Segmentiert das Test-Frame basierend auf dem DMD-basierten Intensitäts-Hintergrund.
    Für jeden Punkt wird der gemessene Intensitätswert mit dem Hintergrundwert (aus bg_intensity)
    verglichen. Liegt die Differenz unter tol * bg_value, wird der Punkt als Hintergrund klassifiziert.

    Parameter:
    - tol: Toleranzfaktor (erhöht von 0.1 auf 0.3)
    """
    pcl = test_frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity'], pcl['ring']), axis=-1)
    else:
        points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity']), axis=-1)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensities = points[:, 3]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        vertical_idx = points[:, 4].astype(int)
    else:
        vertical_idx = get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles)
    # Hintergrundintensität für jeden Punkt (hier wird nur der vertikale Index genutzt)
    bg_vals = bg_intensity[vertical_idx, frame_idx]
    diff = np.abs(intensities - bg_vals)
    background_mask = diff < (tol * bg_vals)
    foreground_mask = ~background_mask
    foreground_points = np.stack((x, y, z), axis=-1)[foreground_mask]
    background_points = np.stack((x, y, z), axis=-1)[background_mask]
    foreground_intensities = intensities[foreground_mask]
    background_intensities = intensities[background_mask]
    return foreground_points, background_points, foreground_intensities, background_intensities


def visualize_point_clouds_intensity_interactive(foreground_points, background_points,
                                                 foreground_intensities, background_intensities,
                                                 FOV_horiz, FOV_vert, azimuth_resltn,
                                                 elevation_resltn, horiz_total_grids, vert_total_grids,
                                                 beam_altitude_angles=None):
    """
    Interaktive Visualisierung basierend auf der intensitätsbasierten Segmentierung.
    Beim Anklicken eines Punktes wird neben Range auch der Intensitätswert ausgegeben.
    """
    fg_len = foreground_points.shape[0]
    bg_len = background_points.shape[0]
    combined_points = np.vstack((foreground_points, background_points)).astype(np.float64)
    combined_intensities = np.concatenate((foreground_intensities, background_intensities))
    colors_fg = np.tile(np.array([[1, 0, 0]], dtype=np.float64), (fg_len, 1))  # Rot für Foreground
    colors_bg = np.tile(np.array([[0, 0, 1]], dtype=np.float64), (bg_len, 1))  # Blau für Background
    combined_colors = np.vstack((colors_fg, colors_bg))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Interaktive Punktwahl: Intensitätsbasiert")
    vis.add_geometry(pcd, reset_bounding_box=True)
    vis.run()  # Auswahl per Shift+Linksklick
    picked_indices = vis.get_picked_points()
    vis.destroy_window()
    if picked_indices is None or len(picked_indices) == 0:
        print("Keine Punkte ausgewählt.")
        return
    combined_np = np.asarray(pcd.points)
    for idx in picked_indices:
        x, y, z = combined_np[idx]
        r_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        intensity_val = combined_intensities[idx]
        hor_idx = get_horizontal_idx(np.array([x]), np.array([y]), FOV_horiz, azimuth_resltn, horiz_total_grids)[0]
        if beam_altitude_angles is not None:
            ver_idx = get_vertical_idx(np.array([z]), np.array([r_val]), FOV_vert, elevation_resltn, vert_total_grids,
                                       beam_altitude_angles)[0]
        else:
            ver_idx = int(np.floor(np.degrees(np.arcsin(z / r_val)) / elevation_resltn)) % vert_total_grids
        group = "Foreground" if idx < fg_len else "Background"
        print(f"Punkt {idx} ({group}): Range = {r_val:.2f}, Intensity = {intensity_val:.2f}")


# ==========================================
# Main-Funktion (angepasst für Intensitätssegmentierung)
# ==========================================
def main():
    # Flags zum Laden vorhandener Modelle und Matrizen
    load_existing_range_model = False
    load_existing_intensity_model = False
    load_existing_range_matrix = True
    load_existing_intensity_matrix = True

    range_model_filename = "../range_thrld_matrix.npy"
    intensity_model_filename = "bg_intensity.npy"
    range_matrix_filename = "../range_matrix.npy"
    intensity_matrix_filename = "intensity_matrix.npy"

    dataset_path = "/mnt/hot_data/dataset/coopscenes/seq_1"
    dataset = Dataloader(dataset_path)
    FOV_vert, FOV_horiz, vert_total_grids, horiz_total_grids, azimuth_resltn, elevation_resltn, beam_altitude_angles = load_parameters(
        dataset)
    total_frame = 4000

    # 1. Lade oder berechne die range_matrix
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
                r, az_idx, el_idx, _ = process_frame(frame, FOV_vert, FOV_horiz, azimuth_resltn,
                                                     elevation_resltn, horiz_total_grids, vert_total_grids,
                                                     beam_altitude_angles)
                update_range_matrix(range_matrix, r, az_idx, el_idx, frame_num)
                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Verarbeitung: {frame_num} Frames verarbeitet.")
            if frame_num >= total_frame:
                break
        print(f"range_matrix Berechnung abgeschlossen. Gesamtanzahl Frames: {frame_num}")
        np.save(range_matrix_filename, range_matrix)

    # 2. Lade oder berechne die intensity_matrix
    if load_existing_intensity_matrix and os.path.exists(intensity_matrix_filename):
        print("Lade vorhandene intensity_matrix ...")
        intensity_matrix = np.load(intensity_matrix_filename)
    else:
        intensity_matrix = np.zeros((vert_total_grids, horiz_total_grids, total_frame), dtype=np.float32)
        frame_num = 0
        for record in dataset:
            for frame in record:
                if frame_num >= total_frame:
                    break
                _, az_idx, el_idx, intensities = process_frame(frame, FOV_vert, FOV_horiz, azimuth_resltn,
                                                               elevation_resltn, horiz_total_grids, vert_total_grids,
                                                               beam_altitude_angles)
                update_intensity_matrix(intensity_matrix, intensities, az_idx, el_idx, frame_num)
                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Verarbeitung: {frame_num} Frames verarbeitet.")
            if frame_num >= total_frame:
                break
        print(f"intensity_matrix Berechnung abgeschlossen. Gesamtanzahl Frames: {frame_num}")
        np.save(intensity_matrix_filename, intensity_matrix)

    # 3. Lade oder trainiere das Range-Hintergrundmodell (range_thrld_matrix)
    if load_existing_range_model and os.path.exists(range_model_filename):
        print("Lade vorhandenes Range-Hintergrundmodell ...")
        range_thrld_matrix = np.load(range_model_filename)
    else:
        range_thrld_matrix = train_background_model(range_matrix, vert_total_grids, horiz_total_grids)
        print("Speichere Range-Hintergrundmodell ...")
        np.save(range_model_filename, range_thrld_matrix)

    # Test-Frame: Hintergrundsubtraktion basierend auf Range
    test_frame = dataset[0][0]
    foreground_points, background_points = background_subtraction(test_frame, range_thrld_matrix,
                                                                  FOV_vert, FOV_horiz,
                                                                  azimuth_resltn, elevation_resltn,
                                                                  horiz_total_grids, vert_total_grids,
                                                                  beam_altitude_angles)
    visualize_point_clouds_interactive(foreground_points, background_points,
                                       FOV_horiz, FOV_vert,
                                       azimuth_resltn, elevation_resltn,
                                       horiz_total_grids, vert_total_grids,
                                       range_thrld_matrix, beam_altitude_angles)

    # 4. DMD-basierte Intensitätsanalyse und Segmentierung:
    # Verwende statt np.mean nun den robusteren Median
    intensity_matrix_avg = np.median(intensity_matrix, axis=1)
    if load_existing_intensity_model and os.path.exists(intensity_model_filename):
        print("Lade vorhandenes Intensitäts-Hintergrundmodell ...")
        bg_intensity = np.load(intensity_model_filename)
    else:
        # Hier kannst du ggf. den Parameter "rank" anpassen (Standard: 1)
        bg_intensity = dmd_background_subtraction(intensity_matrix_avg, rank=1)
        print("Speichere Intensitäts-Hintergrundmodell ...")
        np.save(intensity_model_filename, bg_intensity)

    # Segmentiere den Test-Frame basierend auf Intensität.
    # Hier verwenden wir frame_idx = 0 als Beispiel.
    frame_idx = 0
    fg_points_i, bg_points_i, fg_int_i, bg_int_i = background_subtraction_intensity(
        test_frame, bg_intensity, frame_idx, FOV_vert, FOV_horiz,
        azimuth_resltn, elevation_resltn, horiz_total_grids, vert_total_grids,
        beam_altitude_angles, tol=0.3
    )
    visualize_point_clouds_intensity_interactive(fg_points_i, bg_points_i, fg_int_i, bg_int_i,
                                                 FOV_horiz, FOV_vert, azimuth_resltn,
                                                 elevation_resltn, horiz_total_grids, vert_total_grids,
                                                 beam_altitude_angles)


if __name__ == "__main__":
    main()
