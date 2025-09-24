#!/usr/bin/env python3
"""
Thresholding module for CFTA background subtraction.

This module loads thresholding parameters from a YAML configuration file
and provides functions to compute coarse and fine threshold values
from a sequence of distance measurements.
"""

import numpy as np


def get_farthest_peak(hist: np.ndarray, min_count: int, top_n: int):
    """
    Select the most distant valid peak index from a histogram.

    The function sorts histogram bins by count in descending order,
    then filters the top_n bins to those whose count is at least min_count.
    If any filtered bins remain, returns the largest index among them.
    Otherwise, returns the index of the overall maximum bin.

    Args:
        hist (np.ndarray): 1D array of histogram counts.
        min_count (int): Minimum count threshold to consider a bin valid.
        top_n (int): Number of top bins to inspect.

    Returns:
        tuple[int, list[int]]:
            - Index of the selected peak bin.
            - List of indices corresponding to the top_n bins.
    """
    if hist.size == 0:
        return int(np.argmax(hist)), []

    # Sort bin indices by count descending
    sorted_idx = np.argsort(hist)[::-1]
    top_bins = sorted_idx[:top_n]
    valid = [i for i in top_bins if hist[i] >= min_count]

    if valid:
        sel = int(max(valid))
    else:
        sel = int(sorted_idx[0])

    return sel, top_bins.tolist()


def compute_threshold_meta(distances: np.ndarray, config: dict):
    """
    Compute CFTA thresholds and related metadata from distance measurements.

    Steps:
      1. Noise check: if ratio of non-zero distances < noise_ratio → return dist_max.
      2. Sparse case: if ratio between sparse_ratio_low and sparse_ratio_high,
         check coarse histogram dominance by sparse_static_ratio.
      3. Coarse peak search: build coarse histogram, select peak bin, trim distances.
      4. Fine peak search (Triangle method): refine threshold with fine histogram.

    Args:
        distances (np.ndarray): 1D array of distance measurements (one beam over time).
        config (dict): Thresholding configuration parameters.

    Returns:
        dict: {
            'threshold': float,
            'coarse': {...},
            'fine': {...} (optional)
        }
    """
    # --- Konfiguration einmal am Anfang auslesen ---
    noise_ratio = config.get("noise_ratio", 0.02)
    sparse_ratio_low = config.get("sparse_ratio_low", 0.02)
    sparse_ratio_high = config.get("sparse_ratio_high", 0.30)
    sparse_static_ratio = config.get("sparse_static_ratio", 0.50)

    coarse_top_n = config.get("coarse_top_n", 3)
    coarse_min_ratio = config.get("coarse_min_ratio", 0.20)
    coarse_trim_offset = config.get("coarse_trim_offset", 5.0)

    fine_top_n = config.get("fine_top_n", 5)
    fine_min_ratio = config.get("fine_min_ratio", 0.10)
    fine_bin_num = config.get("fine_bin_num", 200)
    fine_min_bin_size = config.get("fine_min_bin_size", 0.10)

    dist_max = config.get("max_range", 200.0)

    # --- Schritt 1: Noise check ---
    original_size = len(distances)
    nz = distances[distances > 0]
    count = len(nz)
    ratio = count / original_size if original_size > 0 else 0.0

    if ratio < noise_ratio:
        return {"threshold": float(dist_max)}

    # --- Schritt 2: Sparse case ---
    if sparse_ratio_low <= ratio < sparse_ratio_high:
        max_d = np.max(nz)
        bins = max(int(np.ceil(max_d)), 1)
        hist_coarse, edges_coarse = np.histogram(nz, bins=bins, range=(0, max_d))
        peak0 = int(np.argmax(hist_coarse))
        if hist_coarse[peak0] >= sparse_static_ratio * count:
            arr = nz.copy()
        else:
            return {"threshold": float(dist_max)}
    else:
        arr = nz.copy()

    # --- Schritt 3: Coarse peak search ---
    max_d2 = np.max(arr)
    bins2 = max(int(np.ceil(max_d2)), 1)
    hist2, edges2 = np.histogram(arr, bins=bins2, range=(0, max_d2))
    min_count_coarse = int(coarse_min_ratio * len(arr))
    sel_coarse, _ = get_farthest_peak(hist2, min_count=min_count_coarse, top_n=coarse_top_n)
    bin_width = edges2[1] - edges2[0]
    trim_threshold = edges2[sel_coarse] + bin_width / 2 + coarse_trim_offset
    trimmed = arr[arr <= trim_threshold]

    if len(trimmed) < 2:
        return {
            "threshold": float(dist_max),
            "coarse": {
                "hist": hist2,
                "edges": edges2,
                "sel": sel_coarse,
                "trim_thr": float(trim_threshold),
            },
        }

    # --- Schritt 4: Fine histogram + Triangle method ---
    R = np.max(trimmed)
    step = max(R / fine_bin_num, fine_min_bin_size)
    half = step / 2
    binsf = np.arange(-half, R + step, step)
    hist_fine, edges_fine = np.histogram(trimmed, bins=binsf)
    adjusted_hist = hist_fine + 1
    tot = len(trimmed)
    centers = (edges_fine[:-1] + edges_fine[1:]) / 2
    min_count_fine = int(fine_min_ratio * tot)
    sel_fine, _ = get_farthest_peak(adjusted_hist, min_count=min_count_fine, top_n=fine_top_n)

    y0, y1 = adjusted_hist[0], adjusted_hist[sel_fine]
    x0, x1 = 0, sel_fine
    k = (y0 - y1) / ((x0 - x1) + 1e-6)
    a, b, c = k, -1, y1

    distances_line = np.zeros(sel_fine, dtype=float)
    for i in range(sel_fine):
        yi = adjusted_hist[i]
        s = a * i + b * yi + c
        distances_line[i] = (s / np.sqrt(a * a + b * b)) if s >= 0 else 0.0

    valid_idxs = np.where(distances_line > 0)[0]
    level = int(valid_idxs[np.argmax(distances_line[valid_idxs])]) if len(valid_idxs) else 0

    threshold_value = edges_fine[level] + half

    return {
        "threshold": float(threshold_value),
        "coarse": {
            "hist": hist2,
            "edges": edges2,
            "sel": sel_coarse,
            "trim_thr": float(trim_threshold),
        },
        "fine": {
            "hist": adjusted_hist,
            "edges": edges_fine,
            "centers": centers,
            "sel": sel_fine,
        },
    }


def compute_threshold(distances: np.ndarray, config: dict) -> float:
    """
    Compute only the threshold value from distance measurements.

    Args:
        distances (np.ndarray): 1D array of distance measurements.
        dist_max (float): Maximum distance to use as fallback.
        fine_bin_num (int): Number of bins for fine histogram.

    Returns:
        float: Computed threshold distance.
    """
    meta = compute_threshold_meta(distances, config)
    return meta['threshold']


def train_background_model(bg_model: "BackgroundModel"):
    sensor_meta = bg_model.dataset.meta
    range_thrld_matrix = np.ones((sensor_meta.vert_total_grids, sensor_meta.horiz_total_grids), dtype=np.float32) * 200
    for i in range(sensor_meta.vert_total_grids):
        for j in range(sensor_meta.horiz_total_grids):
            distances = bg_model.range_matrix[i, j, :]
            range_thrld_matrix[i, j] = compute_threshold(distances, bg_model.config.thresholding)
        if i % 10 == 0:
            print(f"Background training: processed row {i}/{sensor_meta.vert_total_grids}.")
    print("Background training completed.")
    return range_thrld_matrix
