import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from L1_08_config import get_selected_profile_name


DEFAULT_INPUT_CONFIG_JSON = Path(__file__).resolve().parent.parent / "input_config.json"
QAM_SEED_ENV_VAR = "L1_08_QAM_SEED"


@lru_cache(maxsize=4)
def load_input_config(config_json: Path = DEFAULT_INPUT_CONFIG_JSON) -> dict[str, Any]:
    if not config_json.is_file():
        return {}

    loaded = json.loads(config_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_json} must contain a JSON object.")
    return loaded


def get_available_input_profiles(config_json: Path = DEFAULT_INPUT_CONFIG_JSON) -> list[str]:
    config = load_input_config(config_json)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    return sorted(str(name) for name in profiles)


def get_active_input_config(profile_name: str | None = None, config_json: Path = DEFAULT_INPUT_CONFIG_JSON) -> dict[str, Any]:
    config = load_input_config(config_json)
    active = config.get("active", {})
    if not isinstance(active, dict):
        return {}

    selected_profile = profile_name if profile_name is not None else get_selected_profile_name()
    if selected_profile is None:
        merged = _deep_merge_dict({}, active)
        _apply_input_seed_env_overrides(merged)
        return merged

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Input config selected a profile, but top-level 'profiles' is missing or not an object.")

    profile = profiles.get(selected_profile)
    if not isinstance(profile, dict):
        available = ", ".join(get_available_input_profiles(config_json)) or "none"
        raise ValueError(f"Unknown input profile '{selected_profile}'. Available profiles: {available}.")

    merged = _deep_merge_dict(active, profile)
    _apply_input_seed_env_overrides(merged)
    return merged


def get_input_config_value(section: str, key: str, default: Any, profile_name: str | None = None) -> Any:
    active = get_active_input_config(profile_name=profile_name)
    section_data = active.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge_dict(value, {})
        else:
            merged[key] = value

    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
        elif isinstance(value, dict):
            merged[key] = _deep_merge_dict(value, {})
        else:
            merged[key] = value

    return merged


def _apply_input_seed_env_overrides(config: dict[str, Any]) -> None:
    env_value = os.environ.get(QAM_SEED_ENV_VAR)
    if env_value is None or env_value.strip() == "":
        return

    section = config.setdefault("qam_evm", {})
    if not isinstance(section, dict):
        raise ValueError("Input config section 'qam_evm' must be an object to override its seed.")
    section["seed"] = int(env_value)
