import os
import yaml
import logging
from typing import Any, Dict
import importlib

log = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, path: str = None):
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        cfg_file = path or os.path.join(project_root, "config", "config.yaml")

        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f) or {}

        self.data_format: str = self._cfg.get("data_format", "")
        self.data_path: str = self._cfg.get("data_path") or os.path.join(project_root, "data")
        self.total_frames: int = int(self._cfg.get("total_frames", 1000))

        self.load_background_model: bool = bool(self._cfg.get("load_background_model", False))
        self.model_path: str = self._cfg.get("model_path") or os.path.join(project_root, "data")

        self.thresholding: Dict[str, Any] = self._cfg.get("thresholding", {})

        self._post_processing: Dict[str, Any] = self._cfg.get("post-processing", {})
        self.outlier: Dict[str, Any] = self._post_processing.get("outlier", {})
        self.clustering: Dict[str, Any] = self._post_processing.get("clustering", {})

        log.debug("Loaded config from %s", cfg_file)

    def create_dataset(self):
        """
        Factory: build dataset object dynamically depending on data_format.
        Expects a module 'datasets.<format>' with a class '<Format>' inside.
        Example: data_format=CoopScenes -> datasets.coopscenes.CoopScenes
        """
        fmt = self.data_format.lower()
        class_name = self.data_format  # e.g. "CoopScenes", "A42"

        try:
            mod = importlib.import_module(f"datasets.{fmt}")
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unsupported data_format '{self.data_format}': {e}")

        return cls(self.data_path)
