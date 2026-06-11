import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


def _resolve_path(config_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _regularization_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-", "m")


@dataclass(frozen=True)
class FixedPointFormat:
    total_bits: int
    frac_bits: int
    name: str | None = None

    @property
    def integer_bits_including_sign(self) -> int:
        return self.total_bits - self.frac_bits

    @property
    def label(self) -> str:
        if self.name:
            return self.name.replace(".", "_")
        return f"Q{self.integer_bits_including_sign}_{self.frac_bits}"

    @property
    def display_label(self) -> str:
        if self.name:
            return self.name
        return self.label.replace("_", ".")


@dataclass(frozen=True)
class SeedCase:
    name: str
    h1_seed: int
    behavior_seed: int
    qam_seed: int

    @property
    def label(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, int | str]:
        return {
            "seed_case": self.name,
            "h1_seed": self.h1_seed,
            "behavior_seed": self.behavior_seed,
            "qam_seed": self.qam_seed,
        }


@dataclass(frozen=True)
class SweepCombo:
    profile: str | None
    seed_case: SeedCase | None
    l1_08_tap_num: int
    l1_08_regularization: float
    l1_08_fixed_point: FixedPointFormat
    l1_09_allpass_sections: int
    l1_09_fixed_point: FixedPointFormat

    @property
    def profile_label(self) -> str:
        return self.profile or "active"

    @property
    def folder_name(self) -> str:
        seed_part = f"{self.seed_case.label}_" if self.seed_case is not None else ""
        return (
            f"{self.profile_label}_"
            f"{seed_part}"
            f"l108tap{self.l1_08_tap_num:03d}_"
            f"reg{_regularization_label(self.l1_08_regularization)}_"
            f"l108{self.l1_08_fixed_point.label.lower()}_"
            f"l109sec{self.l1_09_allpass_sections:02d}_"
            f"l109{self.l1_09_fixed_point.label.lower()}"
        )

    def to_dict(self) -> dict[str, Any]:
        seed_data = (
            self.seed_case.to_dict()
            if self.seed_case is not None
            else {"seed_case": "active", "h1_seed": "", "behavior_seed": "", "qam_seed": ""}
        )
        return {
            "profile": self.profile_label,
            **seed_data,
            "l1_08_tap_num": self.l1_08_tap_num,
            "l1_08_regularization": self.l1_08_regularization,
            "l1_08_coeff_total_bits": self.l1_08_fixed_point.total_bits,
            "l1_08_coeff_frac_bits": self.l1_08_fixed_point.frac_bits,
            "l1_08_fixed_format": self.l1_08_fixed_point.display_label,
            "l1_09_allpass_sections": self.l1_09_allpass_sections,
            "l1_09_coeff_total_bits": self.l1_09_fixed_point.total_bits,
            "l1_09_coeff_frac_bits": self.l1_09_fixed_point.frac_bits,
            "l1_09_fixed_format": self.l1_09_fixed_point.display_label,
            # Legacy column names are kept so the existing analyzer can still read the summary CSV.
            "tap_num": self.l1_08_tap_num,
            "regularization": self.l1_08_regularization,
            "coeff_total_bits": self.l1_08_fixed_point.total_bits,
            "coeff_frac_bits": self.l1_08_fixed_point.frac_bits,
            "format": self.l1_08_fixed_point.display_label,
        }


@dataclass(frozen=True)
class OutputConfig:
    group_by_current_seed: bool
    overwrite_existing_combo: bool
    cleanup_sim_outputs_after_copy: bool
    sweep_folder_name: str | None


@dataclass(frozen=True)
class StageConfig:
    run_behavior_simulation: bool
    run_qam_evm_simulation: bool
    run_l1_09: bool
    run_l1_09_evm_lin: bool
    run_l1_09_qam_evm: bool
    l1_09_validation_coeff_mode: str


@dataclass(frozen=True)
class SweepSettings:
    config_path: Path
    repo_root: Path
    l1_08_sim_dir: Path
    l1_09_sim_dir: Path
    l1_08_config: Path
    l1_09_config: Path
    input_config: Path
    output_root: Path
    output: OutputConfig
    stages: StageConfig
    profiles: list[str | None]
    seed_cases: list[SeedCase | None]
    l1_08_tap_nums: list[int]
    l1_08_regularizations: list[float]
    l1_08_fixed_points: list[FixedPointFormat]
    l1_09_allpass_sections: list[int]
    l1_09_fixed_points: list[FixedPointFormat]

    @classmethod
    def from_json(cls, config_path: Path) -> "SweepSettings":
        config_path = config_path.resolve()
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a JSON object.")

        paths = _require_dict(loaded, "paths")
        output = _require_dict(loaded, "output")
        stages = _require_dict(loaded, "stages")
        sweep = _require_dict(loaded, "sweep")
        l1_08 = _require_dict(sweep, "l1_08") if "l1_08" in sweep else sweep
        l1_09 = _require_dict(sweep, "l1_09") if "l1_09" in sweep else {}

        validation_mode = str(stages.get("l1_09_validation_coeff_mode", "both"))
        if validation_mode not in {"float", "fixed", "both"}:
            raise ValueError("stages.l1_09_validation_coeff_mode must be one of: float, fixed, both.")

        return cls(
            config_path=config_path,
            repo_root=_resolve_path(config_path, str(paths.get("repo_root", "."))),
            l1_08_sim_dir=_resolve_path(config_path, str(paths.get("l1_08_sim_dir", paths.get("sim_dir", "L1-08+L1-09_sim_base_plan/L1-08_sim")))),
            l1_09_sim_dir=_resolve_path(config_path, str(paths.get("l1_09_sim_dir", "L1-08+L1-09_sim_base_plan/L1_09_sim"))),
            l1_08_config=_resolve_path(config_path, str(paths.get("l1_08_config", paths.get("base_experiment_config", "config_base_plan.json")))),
            l1_09_config=_resolve_path(config_path, str(paths.get("l1_09_config", "config_base_plan.json"))),
            input_config=_resolve_path(config_path, str(paths.get("input_config", "config_input.json"))),
            output_root=_resolve_path(config_path, str(paths.get("output_root", "sweep_result"))),
            output=OutputConfig(
                group_by_current_seed=bool(output.get("group_by_current_seed", True)),
                overwrite_existing_combo=bool(output.get("overwrite_existing_combo", True)),
                cleanup_sim_outputs_after_copy=bool(output.get("cleanup_sim_outputs_after_copy", False)),
                sweep_folder_name=_optional_non_empty_str(output.get("sweep_folder_name")),
            ),
            stages=StageConfig(
                run_behavior_simulation=bool(stages.get("run_behavior_simulation", True)),
                run_qam_evm_simulation=bool(stages.get("run_qam_evm_simulation", True)),
                run_l1_09=bool(stages.get("run_l1_09", True)),
                run_l1_09_evm_lin=bool(stages.get("run_l1_09_evm_lin", True)),
                run_l1_09_qam_evm=bool(stages.get("run_l1_09_qam_evm", True)),
                l1_09_validation_coeff_mode=validation_mode,
            ),
            profiles=_optional_profile_list(sweep),
            seed_cases=_optional_seed_case_list(sweep),
            l1_08_tap_nums=[int(item) for item in _require_list(l1_08, "tap_num")],
            l1_08_regularizations=[float(item) for item in _require_list(l1_08, "regularization")],
            l1_08_fixed_points=_fixed_point_formats(l1_08),
            l1_09_allpass_sections=[int(item) for item in _require_list(l1_09, "allpass_sections")],
            l1_09_fixed_points=_fixed_point_formats(l1_09),
        )

    def combos(self) -> list[SweepCombo]:
        return [
            SweepCombo(
                profile=profile,
                seed_case=seed_case,
                l1_08_tap_num=tap_num,
                l1_08_regularization=regularization,
                l1_08_fixed_point=l1_08_fixed_point,
                l1_09_allpass_sections=l1_09_sections,
                l1_09_fixed_point=l1_09_fixed_point,
            )
            for profile, seed_case, tap_num, regularization, l1_08_fixed_point, l1_09_sections, l1_09_fixed_point in product(
                self.profiles,
                self.seed_cases,
                self.l1_08_tap_nums,
                self.l1_08_regularizations,
                self.l1_08_fixed_points,
                self.l1_09_allpass_sections,
                self.l1_09_fixed_points,
            )
        ]

    def current_seed_label(self) -> str:
        if any(profile is not None for profile in self.profiles):
            if any(seed_case is not None for seed_case in self.seed_cases):
                return "full_pipeline_bandwidth_profile_seed_sweep"
            if len(self.profiles) == 1:
                return f"full_pipeline_profile_{self.profiles[0] or 'active'}"
            return "full_pipeline_bandwidth_profile_sweep"

        if self.input_config.is_file():
            input_config = json.loads(self.input_config.read_text(encoding="utf-8"))
            input_active = input_config.get("active", {}) if isinstance(input_config, dict) else {}
            h1_seed = input_active.get("h1", {}).get("seed", "none")
            behavior_seed = input_active.get("behavior", {}).get("seed", "none")
            qam_seed = input_active.get("qam_evm", {}).get("seed", "none")
        else:
            h1_seed = behavior_seed = qam_seed = "none"
        return f"full_pipeline_h1_{h1_seed}_behavior_{behavior_seed}_qam_{qam_seed}"

    def sweep_output_dir(self) -> Path:
        if self.output.sweep_folder_name:
            return self.output_root / self.output.sweep_folder_name
        if self.output.group_by_current_seed:
            return self.output_root / self.current_seed_label()
        return self.output_root


# Backward-compatible alias used by older caller code.
SweepSettings.sim_dir = property(lambda self: self.l1_08_sim_dir)
SweepSettings.base_experiment_config = property(lambda self: self.l1_08_config)


def _optional_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config field '{key}' must be an object.")
    return value


def _as_non_empty_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"config field '{field_name}' must not be empty.")
        return value
    if value is None:
        raise ValueError(f"config field '{field_name}' is required.")
    return [value]


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    if key not in data:
        raise ValueError(f"config field '{key}' is required.")
    return _as_non_empty_list(data.get(key), key)


def _optional_profile_list(sweep: dict[str, Any]) -> list[str | None]:
    value = sweep.get("bandwidth_profiles", sweep.get("profiles", [None]))
    if value is None:
        return [None]
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError("config field 'bandwidth_profiles' must be a non-empty list, a string, or null when provided.")

    profiles: list[str | None] = []
    for item in value:
        profiles.append(None if item is None else str(item))
    return profiles


def _optional_seed_case_list(sweep: dict[str, Any]) -> list[SeedCase | None]:
    value = sweep.get("seed_cases", [None])
    if value is None:
        return [None]
    if not isinstance(value, list) or not value:
        raise ValueError("config field 'seed_cases' must be a non-empty list or null when provided.")

    seed_cases: list[SeedCase | None] = []
    for item in value:
        if item is None:
            seed_cases.append(None)
            continue
        if not isinstance(item, dict):
            raise ValueError("each seed_cases item must be an object with name, h1_seed, behavior_seed, qam_seed.")
        seed_cases.append(
            SeedCase(
                name=str(item["name"]),
                h1_seed=int(item["h1_seed"]),
                behavior_seed=int(item["behavior_seed"]),
                qam_seed=int(item["qam_seed"]),
            )
        )
    return seed_cases


def _fixed_point_formats(section: dict[str, Any]) -> list[FixedPointFormat]:
    value = section.get("fixed_point")
    if value is not None:
        if isinstance(value, dict) and "formats" in value:
            value = value["formats"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not value:
            raise ValueError("fixed_point must be an object, a non-empty list, or {'formats': [...]}." )
        return [_parse_fixed_point_item(item) for item in value]

    # Legacy layout: coeff_total_bits plus coeff_frac_bits.
    total_bits_values = _as_non_empty_list(section.get("coeff_total_bits"), "coeff_total_bits")
    frac_bits_values = _as_non_empty_list(section.get("coeff_frac_bits"), "coeff_frac_bits")
    return [
        FixedPointFormat(total_bits=int(total_bits), frac_bits=int(frac_bits))
        for total_bits, frac_bits in product(total_bits_values, frac_bits_values)
    ]


def _parse_fixed_point_item(item: Any) -> FixedPointFormat:
    if not isinstance(item, dict):
        raise ValueError("each fixed_point item must be an object.")
    return FixedPointFormat(
        total_bits=int(item["coeff_total_bits"]),
        frac_bits=int(item["coeff_frac_bits"]),
        name=str(item["name"]) if item.get("name") else None,
    )
