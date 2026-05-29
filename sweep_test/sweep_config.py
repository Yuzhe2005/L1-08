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

    @property
    def integer_bits_including_sign(self) -> int:
        return self.total_bits - self.frac_bits

    @property
    def label(self) -> str:
        return f"Q{self.integer_bits_including_sign}_{self.frac_bits}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "coeff_total_bits": self.total_bits,
            "coeff_frac_bits": self.frac_bits,
            "format": self.label.replace("_", "."),
        }


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
    tap_num: int
    regularization: float
    fixed_point: FixedPointFormat
    profile: str | None = None
    seed_case: SeedCase | None = None

    @property
    def profile_label(self) -> str:
        return self.profile or "active"

    @property
    def folder_name(self) -> str:
        seed_part = f"{self.seed_case.label}_" if self.seed_case is not None else ""
        return (
            f"{self.profile_label}_"
            f"{seed_part}"
            f"tap{self.tap_num:03d}_"
            f"reg{_regularization_label(self.regularization)}_"
            f"{self.fixed_point.label.lower()}"
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
            "tap_num": self.tap_num,
            "regularization": self.regularization,
            **self.fixed_point.to_dict(),
        }


@dataclass(frozen=True)
class OutputConfig:
    group_by_current_seed: bool
    overwrite_existing_combo: bool
    cleanup_sim_outputs_after_copy: bool


@dataclass(frozen=True)
class StageConfig:
    run_behavior_simulation: bool
    run_qam_evm_simulation: bool


@dataclass(frozen=True)
class SweepSettings:
    config_path: Path
    repo_root: Path
    sim_dir: Path
    base_experiment_config: Path
    output_root: Path
    output: OutputConfig
    stages: StageConfig
    profiles: list[str | None]
    seed_cases: list[SeedCase | None]
    tap_nums: list[int]
    regularizations: list[float]
    coeff_total_bits: int
    coeff_frac_bits: list[int]

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

        return cls(
            config_path=config_path,
            repo_root=_resolve_path(config_path, str(paths["repo_root"])),
            sim_dir=_resolve_path(config_path, str(paths["sim_dir"])),
            base_experiment_config=_resolve_path(config_path, str(paths["base_experiment_config"])),
            output_root=_resolve_path(config_path, str(paths["output_root"])),
            output=OutputConfig(
                group_by_current_seed=bool(output.get("group_by_current_seed", True)),
                overwrite_existing_combo=bool(output.get("overwrite_existing_combo", True)),
                cleanup_sim_outputs_after_copy=bool(output.get("cleanup_sim_outputs_after_copy", False)),
            ),
            stages=StageConfig(
                run_behavior_simulation=bool(stages.get("run_behavior_simulation", True)),
                run_qam_evm_simulation=bool(stages.get("run_qam_evm_simulation", True)),
            ),
            profiles=_optional_profile_list(sweep),
            seed_cases=_optional_seed_case_list(sweep),
            tap_nums=[int(item) for item in _require_list(sweep, "tap_num")],
            regularizations=[float(item) for item in _require_list(sweep, "regularization")],
            coeff_total_bits=int(sweep["coeff_total_bits"]),
            coeff_frac_bits=[int(item) for item in _require_list(sweep, "coeff_frac_bits")],
        )

    def combos(self) -> list[SweepCombo]:
        return [
            SweepCombo(
                profile=profile,
                seed_case=seed_case,
                tap_num=tap_num,
                regularization=regularization,
                fixed_point=FixedPointFormat(total_bits=self.coeff_total_bits, frac_bits=frac_bits),
            )
            for profile, seed_case, tap_num, regularization, frac_bits in product(
                self.profiles,
                self.seed_cases,
                self.tap_nums,
                self.regularizations,
                self.coeff_frac_bits,
            )
        ]

    def current_seed_label(self) -> str:
        if any(profile is not None for profile in self.profiles):
            if any(seed_case is not None for seed_case in self.seed_cases):
                return "bandwidth_profile_seed_sweep"
            if len(self.profiles) == 1:
                return f"profile_{self.profiles[0] or 'active'}"
            return "bandwidth_profile_sweep"

        base = json.loads(self.base_experiment_config.read_text(encoding="utf-8"))
        active = base.get("active", {})
        h1_seed = active.get("h1", {}).get("seed", "none")
        behavior_seed = active.get("behavior", {}).get("seed", "none")
        qam_seed = active.get("qam_evm", {}).get("seed", "none")
        return f"h1_{h1_seed}_behavior_{behavior_seed}_qam_{qam_seed}"

    def sweep_output_dir(self) -> Path:
        if self.output.group_by_current_seed:
            return self.output_root / self.current_seed_label()
        return self.output_root


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config field '{key}' must be an object.")
    return value


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"config field '{key}' must be a non-empty list.")
    return value


def _optional_profile_list(sweep: dict[str, Any]) -> list[str | None]:
    value = sweep.get("profiles", [None])
    if value is None:
        return [None]
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError("config field 'profiles' must be a non-empty list, a string, or null when provided.")

    profiles: list[str | None] = []
    for item in value:
        if item is None:
            profiles.append(None)
        else:
            profiles.append(str(item))
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
