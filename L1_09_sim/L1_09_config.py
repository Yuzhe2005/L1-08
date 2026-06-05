import json
from pathlib import Path
from typing import Any


L1_09_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L1_09_ROOT.parent
CONFIG_PATH = REPO_ROOT / "L1_09_experiment_config.json"


def load_l1_09_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"L1-09 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("L1-09 config root must be a JSON object.")
    return config


def get_l1_09_active_section(section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_l1_09_config()
    active = config.get("active", {})
    if not isinstance(active, dict):
        raise ValueError("L1-09 config active section must be an object.")
    value = active.get(section, default if default is not None else {})
    if not isinstance(value, dict):
        raise ValueError(f"L1-09 config active.{section} must be an object.")
    return value


def get_l1_09_config_value(section: str, key: str, default: Any = None) -> Any:
    return get_l1_09_active_section(section).get(key, default)
