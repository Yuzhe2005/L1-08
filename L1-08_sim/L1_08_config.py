import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_JSON = Path(__file__).resolve().parent.parent / "L1_08_experiment_config.json"
PROFILE_ENV_VAR = "L1_08_PROFILE"
SEED_CASE_ENV_VAR = "L1_08_SEED_CASE"
H1_SEED_ENV_VAR = "L1_08_H1_SEED"
BEHAVIOR_SEED_ENV_VAR = "L1_08_BEHAVIOR_SEED"
QAM_SEED_ENV_VAR = "L1_08_QAM_SEED"


@lru_cache(maxsize=4)
def load_l1_08_config(config_json: Path = DEFAULT_CONFIG_JSON) -> dict[str, Any]:
    if not config_json.is_file():
        return {}

    loaded = json.loads(config_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_json} must contain a JSON object.")
    return loaded


def get_selected_profile_name(default: str | None = None) -> str | None:
    profile_name = os.environ.get(PROFILE_ENV_VAR, "").strip()
    return profile_name or default


def get_selected_seed_case_name(default: str | None = None) -> str | None:
    seed_case_name = os.environ.get(SEED_CASE_ENV_VAR, "").strip()
    return seed_case_name or default


def get_available_profiles(config_json: Path = DEFAULT_CONFIG_JSON) -> list[str]:
    config = load_l1_08_config(config_json)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    return sorted(str(name) for name in profiles)


def get_active_config(profile_name: str | None = None, config_json: Path = DEFAULT_CONFIG_JSON) -> dict[str, Any]:
    config = load_l1_08_config(config_json)
    active = config.get("active", {})
    if not isinstance(active, dict):
        return {}

    selected_profile = profile_name if profile_name is not None else get_selected_profile_name()
    if selected_profile is None:
        merged = _deep_merge_dict({}, active)
        _apply_seed_env_overrides(merged)
        return merged

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Config selected a profile, but top-level 'profiles' is missing or not an object.")

    profile = profiles.get(selected_profile)
    if not isinstance(profile, dict):
        available = ", ".join(get_available_profiles(config_json)) or "none"
        raise ValueError(f"Unknown L1-08 profile '{selected_profile}'. Available profiles: {available}.")

    merged = _deep_merge_dict(active, profile)
    _apply_seed_env_overrides(merged)
    return merged


def get_active_config_value(section: str, key: str, default: Any, profile_name: str | None = None) -> Any:
    active = get_active_config(profile_name=profile_name)
    section_data = active.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


def get_common_config_value(key: str, default: Any, profile_name: str | None = None) -> Any:
    return get_active_config_value("common", key, default, profile_name=profile_name)


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


def _apply_seed_env_overrides(config: dict[str, Any]) -> None:
    for section_name, env_var in [
        ("h1", H1_SEED_ENV_VAR),
        ("behavior", BEHAVIOR_SEED_ENV_VAR),
        ("qam_evm", QAM_SEED_ENV_VAR),
    ]:
        env_value = os.environ.get(env_var)
        if env_value is None or env_value.strip() == "":
            continue

        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be an object to override its seed.")
        section["seed"] = int(env_value)
