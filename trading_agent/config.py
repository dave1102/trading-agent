"""Loads config.toml from the project root."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
