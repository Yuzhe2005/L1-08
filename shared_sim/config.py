import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CONFIG_JSON = REPO_ROOT / "config_input.json"
BASE_PLAN_CONFIG_JSON = REPO_ROOT / "config_base_plan.json"
PLAN_B_CONFIG_JSON = REPO_ROOT / "config_plan_b.json"

# Legacy aliases kept for sweep transition references.
EXPERIMENT_CONFIG_JSON = INPUT_CONFIG_JSON
DEFAULT_CONFIG_JSON = INPUT_CONFIG_JSON
DEFAULT_INPUT_CONFIG_JSON = INPUT_CONFIG_JSON

PROFILE_ENV_VAR = "L1_08_PROFILE"
SEED_CASE_ENV_VAR = "L1_08_SEED_CASE"
H1_SEED_ENV_VAR = "L1_08_H1_SEED"
BEHAVIOR_SEED_ENV_VAR = "L1_08_BEHAVIOR_SEED"
QAM_SEED_ENV_VAR = "L1_08_QAM_SEED"
L1_09_ALLPASS_SECTIONS_ENV_VAR = "L1_09_ALLPASS_SECTIONS"
L1_09_COEFF_TOTAL_BITS_ENV_VAR = "L1_09_COEFF_TOTAL_BITS"
L1_09_COEFF_FRAC_BITS_ENV_VAR = "L1_09_COEFF_FRAC_BITS"
L1_09_VALIDATION_COEFF_MODE_ENV_VAR = "L1_09_VALIDATION_COEFF_MODE"
L1_09_SKIP_EVM_LIN_ENV_VAR = "L1_09_SKIP_EVM_LIN"
L1_09_SKIP_QAM_EVM_ENV_VAR = "L1_09_SKIP_QAM_EVM"


@lru_cache(maxsize=4)
def load_input_config(config_json: Path = INPUT_CONFIG_JSON) -> dict[str, Any]:
    return _load_json(config_json)


@lru_cache(maxsize=4)
def load_base_plan_config(config_json: Path = BASE_PLAN_CONFIG_JSON) -> dict[str, Any]:
    return _load_json(config_json)


@lru_cache(maxsize=4)
def load_plan_b_config(config_json: Path = PLAN_B_CONFIG_JSON) -> dict[str, Any]:
    return _load_json(config_json)


def load_experiment_config(config_json: Path = INPUT_CONFIG_JSON) -> dict[str, Any]:
    """Backward-compatible snapshot: input config file contents."""
    return load_input_config(config_json)


def selected_profile(default: str | None = None) -> str | None:
    profile_name = os.environ.get(PROFILE_ENV_VAR, "").strip()
    if profile_name:
        return profile_name
    config = load_input_config()
    selected = config.get("selected_profile")
    if selected is None:
        return default
    selected_text = str(selected).strip()
    return selected_text or default


def get_selected_profile_name(default: str | None = None) -> str | None:
    return selected_profile(default=default)


def get_selected_seed_case_name(default: str | None = None) -> str | None:
    seed_case_name = os.environ.get(SEED_CASE_ENV_VAR, "").strip()
    return seed_case_name or default


def get_available_profiles(config_json: Path = INPUT_CONFIG_JSON) -> list[str]:
    config = _load_json(config_json)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    return sorted(str(name) for name in profiles)


def get_available_input_profiles(config_json: Path = INPUT_CONFIG_JSON) -> list[str]:
    return get_available_profiles(config_json)


def input_active(profile_name: str | None = None, config_json: Path = INPUT_CONFIG_JSON) -> dict[str, Any]:
    config = _load_json(config_json)
    active = config.get("active", {})
    if not isinstance(active, dict):
        return {}

    selected = profile_name if profile_name is not None else selected_profile()
    if selected is None:
        merged = _deep_merge_dict({}, active)
        _apply_input_seed_env_overrides(merged)
        return merged

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Config selected a profile, but top-level 'profiles' is missing or not an object.")

    profile = profiles.get(selected)
    if not isinstance(profile, dict):
        available = ", ".join(get_available_profiles(config_json)) or "none"
        raise ValueError(f"Unknown input profile '{selected}'. Available profiles: {available}.")

    merged = _deep_merge_dict(active, profile)
    _apply_input_seed_env_overrides(merged)
    return merged


def input_value(section: str, key: str, default: Any, profile_name: str | None = None) -> Any:
    active = input_active(profile_name=profile_name)
    section_data = active.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


def base_active(config_json: Path = BASE_PLAN_CONFIG_JSON) -> dict[str, Any]:
    config = _load_json(config_json)
    active = config.get("active", {})
    if not isinstance(active, dict):
        return {}
    return _deep_merge_dict({}, active)


def base_value(dotted_section: str, key: str, default: Any, config_json: Path = BASE_PLAN_CONFIG_JSON) -> Any:
    active = base_active(config_json=config_json)
    section_parts = dotted_section.split(".")
    section_data: Any = active
    for part in section_parts:
        if not isinstance(section_data, dict):
            return default
        section_data = section_data.get(part, {})
    if not isinstance(section_data, dict):
        return default
    value = section_data.get(key, default)
    return _apply_base_run_env_override(dotted_section, key, value)


def plan_b_active(config_json: Path = PLAN_B_CONFIG_JSON) -> dict[str, Any]:
    config = _load_json(config_json)
    active = config.get("active", {})
    if not isinstance(active, dict):
        return {}
    return _deep_merge_dict({}, active)


def plan_b_value(section: str, key: str, default: Any, config_json: Path = PLAN_B_CONFIG_JSON) -> Any:
    active = plan_b_active(config_json=config_json)
    section_data = active.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


# Backward-compatible aliases used across the codebase.
def get_active_config(profile_name: str | None = None, config_json: Path = INPUT_CONFIG_JSON) -> dict[str, Any]:
    return input_active(profile_name=profile_name, config_json=config_json)


def get_active_input_config(profile_name: str | None = None, config_json: Path = INPUT_CONFIG_JSON) -> dict[str, Any]:
    return input_active(profile_name=profile_name, config_json=config_json)


def get_common_config_value(key: str, default: Any, profile_name: str | None = None) -> Any:
    return input_value("common", key, default, profile_name=profile_name)


def get_active_config_value(section: str, key: str, default: Any, profile_name: str | None = None) -> Any:
    if section in {"h2_fir", "fixed_point", "run"} or section.startswith("l1_09"):
        dotted = section if "." in section else section
        if section.startswith("l1_09"):
            return base_value(dotted, key, default)
        return base_value(section, key, default)
    return input_value(section, key, default, profile_name=profile_name)


def get_input_config_value(section: str, key: str, default: Any, profile_name: str | None = None) -> Any:
    return input_value(section, key, default, profile_name=profile_name)


def get_l1_09_config_value(section: str, key: str, default: Any = None) -> Any:
    env_overrides: dict[tuple[str, str], tuple[str, type]] = {
        ("allpass", "sections"): (L1_09_ALLPASS_SECTIONS_ENV_VAR, int),
        ("fixed_point", "coeff_total_bits"): (L1_09_COEFF_TOTAL_BITS_ENV_VAR, int),
        ("fixed_point", "coeff_frac_bits"): (L1_09_COEFF_FRAC_BITS_ENV_VAR, int),
    }
    override = env_overrides.get((section, key))
    if override is not None:
        env_var, caster = override
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            return caster(env_value)
    return base_value(f"l1_09.{section}", key, default)


def _apply_base_run_env_override(dotted_section: str, key: str, value: Any) -> Any:
    if dotted_section != "run":
        return value

    env_map: dict[str, tuple[str, type]] = {
        "validation_coeff_mode": (L1_09_VALIDATION_COEFF_MODE_ENV_VAR, str),
        "skip_evm_lin": (L1_09_SKIP_EVM_LIN_ENV_VAR, _env_bool),
        "skip_l1_09_qam_evm": (L1_09_SKIP_QAM_EVM_ENV_VAR, _env_bool),
    }
    override = env_map.get(key)
    if override is None:
        return value

    env_var, caster = override
    env_value = os.environ.get(env_var, "").strip()
    if not env_value:
        return value
    return caster(env_value)


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_json(config_json: Path) -> dict[str, Any]:
    if not config_json.is_file():
        return {}

    loaded = json.loads(config_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_json} must contain a JSON object.")
    return loaded


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge_dict(value, {})
        else:
            merged[key] = value

    for key, value in override.items():
        base_value_item = merged.get(key)
        if isinstance(base_value_item, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base_value_item, value)
        elif isinstance(value, dict):
            merged[key] = _deep_merge_dict(value, {})
        else:
            merged[key] = value

    return merged


def _apply_input_seed_env_overrides(config: dict[str, Any]) -> None:
    for section_name, env_var in [
        ("h1", H1_SEED_ENV_VAR),
        ("behavior", BEHAVIOR_SEED_ENV_VAR),
    ]:
        env_value = os.environ.get(env_var)
        if env_value is None or env_value.strip() == "":
            continue

        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be an object to override its seed.")
        section["seed"] = int(env_value)

    env_value = os.environ.get(QAM_SEED_ENV_VAR)
    if env_value is None or env_value.strip() == "":
        return

    section = config.setdefault("qam_evm", {})
    if not isinstance(section, dict):
        raise ValueError("Input config section 'qam_evm' must be an object to override its seed.")
    section["seed"] = int(env_value)
