#!/usr/bin/env python3
"""
Module: background_model.py

Contains functions to train a background model (threshold matrix)
and perform background/foreground separation on new frames.
"""

import numpy as np
import os
from range_intensity import ConfigManager, create_dataset
from range_intensity.datasets import CoopScenes
from range_intensity.core import train_background_model, build_range_matrix
from typing import Optional


class BackgroundModel:
    def __init__(self) -> None:
        self.config: ConfigManager = ConfigManager()
        self.dataset: Optional[CoopScenes] = create_dataset(self.config.data_format, self.config.data_path)

        self.range_matrix: Optional[np.ndarray] = None
        self.threshold_matrix: Optional[np.ndarray] = None

        if self.config.load_background_model and os.path.exists(self.config.model_path):
            print(f"Loading existing BackgroundModel from {self.config.model_path} ...")
            self._load()
        else:
            print("Building range matrix from scratch...")
            self.range_matrix = build_range_matrix(self.dataset, self.config)

            print("Training background model (Threshold-Matrix)...")
            self.threshold_matrix = train_background_model(self)

            print(f"Saving BackgroundModel to {self.config.model_path} ...")
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        np.savez_compressed(
            self.config.model_path,
            range_matrix=self.range_matrix,
            threshold_matrix=self.threshold_matrix
        )

    def _load(self) -> None:
        data = np.load(self.config.model_path, allow_pickle=True)
        self.range_matrix = data["range_matrix"]
        self.threshold_matrix = data["threshold_matrix"]
