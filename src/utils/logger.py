import logging
import logging.config
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logger(config_path: str = "config/logging_config.yaml") -> logging.Logger:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"Logging configuration must be a YAML mapping: {path}")
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger("app")


logger = setup_logger()
