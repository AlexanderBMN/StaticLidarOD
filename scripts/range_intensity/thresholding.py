#!/usr/bin/env python3
"""
Thresholding module for CFTA background subtraction.

This module loads thresholding parameters from a YAML configuration file
and provides functions to compute coarse and fine threshold values
from a sequence of distance measurements.
"""

import os

import yaml
import numpy as np


def load_config(path="config.yaml"):
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Load thresholding parameters from the "thresholding" section of config.yaml
_cfg = load_config("../../config/config.yaml")["thresholding"]

NOISE_RATIO = _cfg.get("noise_ratio", 0.02)
SPARSE_RATIO_LOW = _cfg.get("sparse_ratio_low", 0.02)
SPARSE_RATIO_HIGH = _cfg.get("sparse_ratio_high", 0.30)
SPARSE_STATIC_RATIO = _cfg.get("sparse_static_ratio", 0.50)

COARSE_TOP_N = _cfg.get("coarse_top_n", 3)
COARSE_MIN_RATIO = _cfg.get("coarse_min_ratio", 0.20)
COARSE_TRIM_OFFSET = _cfg.get("coarse_trim_offset", 5.0)

FINE_TOP_N = _cfg.get("fine_top_n", 5)
FINE_MIN_RATIO = _cfg.get("fine_min_ratio", 0.10)
FINE_BIN_NUM = _cfg.get("fine_bin_num", 200)
FINE_MIN_BIN_SIZE = _cfg.get("fine_min_bin_size", 0.10)

MAX_RANGE = _cfg.get("max_range", 200.0)


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


def compute_threshold_meta(distances: np.ndarray,
                           dist_max: float = MAX_RANGE,
                           fine_bin_num: int = FINE_BIN_NUM):
    """
    Compute CFTA thresholds and related metadata from distance measurements.

    The algorithm proceeds in four steps:
      1. Noise check: if the ratio of non-zero distances is below NOISE_RATIO,
         return dist_max immediately.
      2. Sparse case: if the ratio of non-zero distances lies between
         SPARSE_RATIO_LOW and SPARSE_RATIO_HIGH, build a coarse histogram
         to check if the largest bin is dominant by SPARSE_STATIC_RATIO.
         If not dominant, return dist_max.
      3. Coarse peak search: build a coarse histogram, select the peak bin
         among the top COARSE_TOP_N bins with count ≥ COARSE_MIN_RATIO * count,
         and compute a trim threshold by adding COARSE_TRIM_OFFSET. Trim distances
         above this threshold.
      4. Fine peak search (Triangle method): build a fine histogram on trimmed
         distances, select a peak among the top FINE_TOP_N bins with count ≥
         FINE_MIN_RATIO * total, then compute the largest vertical distance below
         the line connecting the first bin and the selected peak bin to determine
         the final threshold.

    Args:
        distances (np.ndarray): 1D array of distance measurements (one beam over time).
        dist_max (float): Maximum distance to use as fallback threshold.
        fine_bin_num (int): Number of bins for the fine-resolution histogram.

    Returns:
        dict: A dictionary with keys:
            - 'threshold' (float): Final threshold distance.
            - 'coarse' (dict): Metadata for the coarse step, with keys:
                'hist' (np.ndarray), 'edges' (np.ndarray), 'sel' (int), 'trim_thr' (float).
            - 'fine' (dict, optional): Metadata for the fine step, with keys:
                'hist' (np.ndarray), 'edges' (np.ndarray), 'centers' (np.ndarray), 'sel' (int).
              The 'fine' key is omitted if a valid fine step cannot be computed.

    """
    original_size = len(distances)
    nz = distances[distances > 0]
    count = len(nz)
    ratio = count / original_size if original_size > 0 else 0.0

    # 1) Noise check
    if ratio < NOISE_RATIO:
        return {'threshold': float(dist_max)}

    # 2) Sparse case
    if SPARSE_RATIO_LOW <= ratio < SPARSE_RATIO_HIGH:
        max_d = np.max(nz)
        bins = max(int(np.ceil(max_d)), 1)
        hist_coarse, edges_coarse = np.histogram(nz, bins=bins, range=(0, max_d))
        peak0 = int(np.argmax(hist_coarse))
        if hist_coarse[peak0] >= SPARSE_STATIC_RATIO * count:
            arr = nz.copy()
        else:
            return {'threshold': float(dist_max)}
    else:
        arr = nz.copy()

    # 3) Coarse peak search
    max_d2 = np.max(arr)
    bins2 = max(int(np.ceil(max_d2)), 1)
    hist2, edges2 = np.histogram(arr, bins=bins2, range=(0, max_d2))
    min_count_coarse = int(COARSE_MIN_RATIO * len(arr))
    sel_coarse, _ = get_farthest_peak(hist2, min_count=min_count_coarse, top_n=COARSE_TOP_N)
    bin_width = edges2[1] - edges2[0]
    trim_threshold = edges2[sel_coarse] + bin_width / 2 + COARSE_TRIM_OFFSET
    trimmed = arr[arr <= trim_threshold]

    if len(trimmed) < 2:
        return {
            'threshold': float(dist_max),
            'coarse': {
                'hist': hist2,
                'edges': edges2,
                'sel': sel_coarse,
                'trim_thr': float(trim_threshold)
            }
        }

    # 4) Fine histogram and Triangle method
    R = np.max(trimmed)
    step = max(R / fine_bin_num, FINE_MIN_BIN_SIZE)
    half = step / 2
    binsf = np.arange(-half, R + step, step)
    hist_fine, edges_fine = np.histogram(trimmed, bins=binsf)
    adjusted_hist = hist_fine + 1
    tot = len(trimmed)
    centers = (edges_fine[:-1] + edges_fine[1:]) / 2
    min_count_fine = int(FINE_MIN_RATIO * tot)
    sel_fine, _ = get_farthest_peak(adjusted_hist, min_count=min_count_fine, top_n=FINE_TOP_N)

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
        'threshold': float(threshold_value),
        'coarse': {
            'hist': hist2,
            'edges': edges2,
            'sel': sel_coarse,
            'trim_thr': float(trim_threshold)
        },
        'fine': {
            'hist': adjusted_hist,
            'edges': edges_fine,
            'centers': centers,
            'sel': sel_fine
        }
    }


def compute_threshold(distances: np.ndarray,
                      dist_max: float = MAX_RANGE,
                      fine_bin_num: int = FINE_BIN_NUM) -> float:
    """
    Compute only the threshold value from distance measurements.

    Args:
        distances (np.ndarray): 1D array of distance measurements.
        dist_max (float): Maximum distance to use as fallback.
        fine_bin_num (int): Number of bins for fine histogram.

    Returns:
        float: Computed threshold distance.
    """
    meta = compute_threshold_meta(distances, dist_max=dist_max, fine_bin_num=fine_bin_num)
    return meta['threshold']
