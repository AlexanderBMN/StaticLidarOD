import importlib


def create_dataset(data_format: str, data_path: str):
    """
    Dynamically create a dataset object.
    Convention:
      - Module path: range_intensity.datasets.<format_lowercase>
      - Class name: <Format>
    Example:
      data_format="CoopScenes" -> range_intensity.datasets.coop_scenes.CoopScenes
    """
    fmt = data_format.lower()

    try:
        mod = importlib.import_module(f"range_intensity.datasets.{fmt}")
        cls = getattr(mod, data_format)  # expects class name = "CoopScenes"
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported dataset format '{data_format}': {e}")

    return cls(data_path)
