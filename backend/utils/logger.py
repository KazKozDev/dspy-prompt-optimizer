from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: Path) -> None:
    """Configure logging from a YAML config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid logging config: {config_path}")

    log_file = config.get("handlers", {}).get("file", {}).get("filename")
    if isinstance(log_file, str):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance."""
    return logging.getLogger(name)


def safe_setup_default_logging() -> None:
    """Set up a basic logging configuration if none exists."""
    root = logging.getLogger()
    if root.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
