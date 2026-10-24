from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model_config.yaml"


def load_model_config(config_path: str | Path = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    """Load model configuration from YAML, independent of the current directory."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Model configuration must be a YAML mapping: {path}")
    return config
