import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from shared_sim.h1_common import FrequencyGridConfig, H1
from shared_sim.magnitude.H1_edge_rolloff_random_generator import EdgeRolloffRandomConfig, H1EdgeRolloffRandomGenerator
from shared_sim.magnitude.H1_measurement_noise_random_generator import H1MeasurementNoiseRandomGenerator, MeasurementNoiseRandomConfig
from shared_sim.magnitude.H1_notch_bump_random_generator import H1NotchBumpRandomGenerator, NotchBumpRandomConfig
from shared_sim.magnitude.H1_ripple_random_generator import H1RippleRandomGenerator, RippleRandomConfig
from shared_sim.magnitude.H1_slope_random_generator import H1SlopeRandomGenerator, SlopeRandomConfig
from shared_sim.magnitude.H_magnitude_plotter import HMagnitudePlotter
from shared_sim.phase.H1_group_delay_ripple_random_generator import GroupDelayRippleRandomConfig, H1GroupDelayRippleRandomGenerator
from shared_sim.phase.H1_linear_phase_delay_random_generator import H1LinearPhaseDelayRandomGenerator, LinearPhaseDelayRandomConfig
from shared_sim.phase.H1_local_phase_distortion_random_generator import (
    H1LocalPhaseDistortionRandomGenerator,
    LocalPhaseDistortionRandomConfig,
)
from shared_sim.phase.H1_phase_noise_random_generator import H1PhaseNoiseRandomGenerator, PhaseNoiseRandomConfig
from shared_sim.phase.H1_phase_ripple_random_generator import H1PhaseRippleRandomGenerator, PhaseRippleRandomConfig
from shared_sim.phase.H_phase_plotter import HPhasePlotter
from shared_sim.config import (
    input_active,
    input_value,
    selected_profile,
    get_selected_seed_case_name,
)
from shared_sim.io_utils import BASE_RUN_NAME_PREFIX, PLAN_B_RUN_NAME_PREFIX, h1_data_dir
from shared_sim.paths import DATA_ROOT, RESULTS_ROOT
from shared_sim.run_summary import update_run_summary


DEFAULT_RUN_NAME_PREFIX = BASE_RUN_NAME_PREFIX


@dataclass(frozen=True)
class FullCombinedH1Run:
    run_name: str
    data_dir: Path
    graph_dir: Path
    profile: str | None
    magnitude_features: list[H1]
    phase_features: list[H1]
    magnitude_combined: H1
    phase_combined: H1
    together: H1


class H1FullCombinedRandomGenerator:
    def __init__(self, seed: int | None = None, profile: str | None = None) -> None:
        self.profile = profile if profile is not None else selected_profile()
        self.rng = np.random.default_rng(seed)
        self.h1_random_model = _load_h1_random_model_config(self.profile)
        self.grid_config = _make_frequency_grid_config(self.h1_random_model)

    def generate(
        self,
        run_name: str | None = None,
        run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
    ) -> FullCombinedH1Run:
        run_name = run_name or f"{run_name_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_dir = DATA_ROOT / run_name
        graph_dir = RESULTS_ROOT / run_name
        output_data_dir = h1_data_dir(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        output_data_dir.mkdir(parents=True, exist_ok=True)

        magnitude_features = [
            self._generate_magnitude_feature("slope", H1SlopeRandomGenerator, SlopeRandomConfig),
            self._generate_magnitude_feature("ripple", H1RippleRandomGenerator, RippleRandomConfig),
            self._generate_magnitude_feature("notch_bump", H1NotchBumpRandomGenerator, NotchBumpRandomConfig),
            self._generate_magnitude_feature("edge_rolloff", H1EdgeRolloffRandomGenerator, EdgeRolloffRandomConfig),
            self._generate_magnitude_feature(
                "measurement_noise",
                H1MeasurementNoiseRandomGenerator,
                MeasurementNoiseRandomConfig,
            ),
        ]
        phase_features = [
            self._generate_phase_feature(
                "linear_phase_delay",
                H1LinearPhaseDelayRandomGenerator,
                LinearPhaseDelayRandomConfig,
            ),
            self._generate_phase_feature("phase_ripple", H1PhaseRippleRandomGenerator, PhaseRippleRandomConfig),
            self._generate_phase_feature(
                "local_phase_distortion",
                H1LocalPhaseDistortionRandomGenerator,
                LocalPhaseDistortionRandomConfig,
            ),
            self._generate_phase_feature(
                "group_delay_ripple",
                H1GroupDelayRippleRandomGenerator,
                GroupDelayRippleRandomConfig,
            ),
            self._generate_phase_feature("phase_noise", H1PhaseNoiseRandomGenerator, PhaseNoiseRandomConfig),
        ]

        magnitude_combined = self._combine(magnitude_features, "magnitude_combined")
        phase_combined = self._enforce_physical_phase(self._combine(phase_features, "phase_combined"), "phase_combined")
        together = magnitude_combined.add(phase_combined, name="together")

        magnitude_combined.save_csv(output_data_dir / "magnitude_combined.csv")
        phase_combined.save_csv(output_data_dir / "phase_combined.csv")
        together.save_csv(output_data_dir / "together.csv")

        return FullCombinedH1Run(
            run_name=run_name,
            data_dir=data_dir,
            graph_dir=graph_dir,
            profile=self.profile,
            magnitude_features=magnitude_features,
            phase_features=phase_features,
            magnitude_combined=magnitude_combined,
            phase_combined=phase_combined,
            together=together,
        )

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, np.iinfo(np.uint32).max))

    def _generate_magnitude_feature(self, feature_name: str, generator_type: type, config_type: type) -> H1:
        config = _make_feature_config(
            self.h1_random_model,
            group_name="magnitude",
            feature_name=feature_name,
            config_type=config_type,
            grid_config=self.grid_config,
        )
        generated = generator_type(config=config, seed=self._next_seed()).generate(name=feature_name)
        return H1(
            name=feature_name,
            freq_hz=generated.freq_hz,
            h_db=generated.h_db,
            phase_rad=np.zeros_like(generated.h_db),
        )

    def _generate_phase_feature(self, feature_name: str, generator_type: type, config_type: type) -> H1:
        config = _make_feature_config(
            self.h1_random_model,
            group_name="phase",
            feature_name=feature_name,
            config_type=config_type,
            grid_config=self.grid_config,
        )
        generated = generator_type(config=config, seed=self._next_seed()).generate(name=feature_name)
        return H1(
            name=feature_name,
            freq_hz=generated.freq_hz,
            h_db=np.zeros_like(generated.h_db),
            phase_rad=generated.phase_rad,
        )

    def _combine(self, features: list[H1], name: str) -> H1:
        combined = features[0]
        for feature in features[1:]:
            combined = combined.add(feature, name=name)
        return H1(
            name=name,
            freq_hz=combined.freq_hz,
            h_db=combined.h_db,
            phase_rad=combined.phase_rad,
        )

    def _enforce_physical_phase(self, h1: H1, name: str) -> H1:
        return _enforce_positive_group_delay_phase(h1, name=name)


def group_delay_ns(freq_hz: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    omega_rad = 2.0 * np.pi * freq_hz
    return -np.gradient(np.unwrap(phase_rad), omega_rad) * 1e9


def _enforce_positive_group_delay_phase(
    h1: H1,
    name: str,
    min_mean_group_delay_ns: float = 0.20,
    min_low_percentile_group_delay_ns: float = 0.02,
    low_percentile: float = 2.0,
) -> H1:
    group_delay = group_delay_ns(h1.freq_hz, h1.phase_rad)
    mean_delay_ns = float(np.mean(group_delay))
    low_delay_ns = float(np.percentile(group_delay, low_percentile))

    extra_delay_ns = max(
        0.0,
        min_mean_group_delay_ns - mean_delay_ns,
        min_low_percentile_group_delay_ns - low_delay_ns,
    )
    if extra_delay_ns <= 0.0:
        return H1(name=name, freq_hz=h1.freq_hz, h_db=h1.h_db, phase_rad=h1.phase_rad)

    extra_delay_s = extra_delay_ns * 1e-9
    freq_offset_hz = h1.freq_hz - h1.freq_hz[0]
    physical_phase_rad = np.unwrap(h1.phase_rad - 2.0 * np.pi * freq_offset_hz * extra_delay_s)
    return H1(name=name, freq_hz=h1.freq_hz, h_db=h1.h_db, phase_rad=physical_phase_rad)


def _load_h1_random_model_config(profile: str | None = None) -> dict[str, Any]:
    active = input_active(profile_name=profile)
    model = active.get("h1_random_model", {})
    return model if isinstance(model, dict) else {}


def _make_frequency_grid_config(model: dict[str, Any]) -> FrequencyGridConfig:
    default = FrequencyGridConfig()
    section = model.get("frequency_grid", {})
    if not isinstance(section, dict):
        section = {}

    return FrequencyGridConfig(
        f_min_hz=float(section.get("f_min_hz", default.f_min_hz)),
        f_max_hz=float(section.get("f_max_hz", default.f_max_hz)),
        num_points=int(section.get("num_points", default.num_points)),
    )


def _make_feature_config(
    model: dict[str, Any],
    group_name: str,
    feature_name: str,
    config_type: type,
    grid_config: FrequencyGridConfig,
) -> Any:
    default = config_type()
    group = model.get(group_name, {})
    if not isinstance(group, dict):
        group = {}
    section = group.get(feature_name, {})
    if not isinstance(section, dict):
        section = {}

    values: dict[str, Any] = {}
    for field in fields(default):
        if field.name == "grid":
            values[field.name] = grid_config
            continue
        default_value = getattr(default, field.name)
        values[field.name] = _coerce_config_value(section.get(field.name, default_value), default_value)
    return config_type(**values)


def _coerce_config_value(value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    return value


def plot_run(run: FullCombinedH1Run) -> list[Path]:
    plot_dir = run.graph_dir / "h1_full_combined_random"
    h1_dir = h1_data_dir(run.data_dir)
    magnitude_plotter = HMagnitudePlotter(graph_dir=plot_dir)
    phase_plotter = HPhasePlotter(graph_dir=plot_dir)

    plot_paths: list[Path] = []
    plot_paths.append(magnitude_plotter.plot_csv(h1_dir / "magnitude_combined.csv"))
    plot_paths.append(phase_plotter.plot_csv(h1_dir / "phase_combined.csv"))
    return plot_paths


def run_h1_generation(
    seed: int | None = None,
    profile: str | None = None,
    run_name: str | None = None,
    run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
) -> FullCombinedH1Run:
    profile_name = profile if profile is not None else selected_profile()
    seed_case_name = get_selected_seed_case_name()
    if seed is None:
        h1_seed_config = input_value("h1", "seed", None, profile_name=profile_name)
        seed = None if h1_seed_config is None else int(h1_seed_config)

    generator = H1FullCombinedRandomGenerator(seed=seed, profile=profile_name)
    run = generator.generate(run_name=run_name, run_name_prefix=run_name_prefix)
    plot_paths = plot_run(run)
    h1_dir = h1_data_dir(run.data_dir)
    summary_path = update_run_summary(
        run.data_dir,
        "h1_generation",
        {
            "run_name": run.run_name,
            "profile": run.profile or "active",
            "seed_case": seed_case_name or "active",
            "seed": seed,
            "data_dir": run.data_dir,
            "graph_dir": run.graph_dir,
            "frequency": {
                "points": run.magnitude_combined.freq_hz.size,
                "f_min_hz": run.magnitude_combined.freq_hz[0],
                "f_max_hz": run.magnitude_combined.freq_hz[-1],
            },
            "magnitude_features": [
                {
                    "name": feature.name,
                    "ripple_pp_db": feature.ripple_pp_db(),
                }
                for feature in run.magnitude_features
            ],
            "phase_features": [
                {
                    "name": feature.name,
                    "phase_min_rad": np.min(feature.phase_rad),
                    "phase_max_rad": np.max(feature.phase_rad),
                }
                for feature in run.phase_features
            ],
            "magnitude_combined_ripple_pp_db": run.magnitude_combined.ripple_pp_db(),
            "phase_combined_min_rad": np.min(run.phase_combined.phase_rad),
            "phase_combined_max_rad": np.max(run.phase_combined.phase_rad),
            "phase_group_delay_mean_ns": float(np.mean(group_delay_ns(run.phase_combined.freq_hz, run.phase_combined.phase_rad))),
            "phase_group_delay_positive_ratio": float(
                np.mean(group_delay_ns(run.phase_combined.freq_hz, run.phase_combined.phase_rad) > 0.0)
            ),
            "outputs": {
                "magnitude_combined_csv": h1_dir / "magnitude_combined.csv",
                "phase_combined_csv": h1_dir / "phase_combined.csv",
                "together_csv": h1_dir / "together.csv",
                "plots": plot_paths,
            },
        },
        graph_dir=run.graph_dir,
    )

    print(f"run_name: {run.run_name}")
    print(f"profile: {run.profile or 'active'}")
    print(f"seed_case: {seed_case_name or 'active'}")
    print(f"h1_seed: {seed}")
    print(f"data_folder: {run.data_dir}")
    print(f"graph_folder: {run.graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"csv_count: {len(list(h1_dir.glob('*.csv')))}")
    print(f"plot_count: {len(plot_paths)}")
    print(f"magnitude_combined_ripple_pp_db: {run.magnitude_combined.ripple_pp_db():.6f}")
    print(
        "phase_combined_range_rad: "
        f"{np.min(run.phase_combined.phase_rad):.6f} to {np.max(run.phase_combined.phase_rad):.6f}"
    )
    phase_group_delay = group_delay_ns(run.phase_combined.freq_hz, run.phase_combined.phase_rad)
    print(f"phase_group_delay_mean_ns: {np.mean(phase_group_delay):.6f}")
    print(f"phase_group_delay_positive_ratio: {np.mean(phase_group_delay > 0.0):.6f}")
    print(f"together_csv: {h1_dir / 'together.csv'}")
    print("saved_plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
    return run


def parse_h1_cli_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Generate shared H1 magnitude/phase source.")
    parser.add_argument(
        "--run-name-prefix",
        default=BASE_RUN_NAME_PREFIX,
        help=f"Run folder prefix under data/ and graph/. Default: {BASE_RUN_NAME_PREFIX}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_h1_cli_args()
    run_h1_generation(profile=selected_profile(), run_name_prefix=cli_args.run_name_prefix)
