from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "L1-08_sim"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from H1_common import H1
from H1_group_delay_ripple_random_generator import H1GroupDelayRippleRandomGenerator
from H1_linear_phase_delay_random_generator import H1LinearPhaseDelayRandomGenerator
from H1_local_phase_distortion_random_generator import H1LocalPhaseDistortionRandomGenerator
from H1_phase_noise_random_generator import H1PhaseNoiseRandomGenerator
from H1_phase_ripple_random_generator import H1PhaseRippleRandomGenerator
from H_phase_plotter import HPhasePlotter


@dataclass(frozen=True)
class PhysicalPhaseConstraintConfig:
    min_mean_group_delay_ns: float = 0.20
    min_low_percentile_group_delay_ns: float = 0.02
    low_percentile: float = 2.0


@dataclass(frozen=True)
class PhaseCombinedH1Run:
    run_name: str
    data_dir: Path
    results_dir: Path
    single_features: list[H1]
    combined: H1
    group_delay_mean_ns: float
    group_delay_positive_ratio: float


class H1PhaseCombinedRandomGenerator:
    def __init__(
        self,
        seed: int | None = None,
        physical_constraints: PhysicalPhaseConstraintConfig | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.physical_constraints = physical_constraints or PhysicalPhaseConstraintConfig()

    def generate(self, run_name: str | None = None) -> PhaseCombinedH1Run:
        run_name = run_name or f"h1_phase_combined_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data" / run_name
        results_dir = project_root / "results" / run_name
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        features = [
            self._generate_feature("linear_phase_delay", H1LinearPhaseDelayRandomGenerator),
            self._generate_feature("phase_ripple", H1PhaseRippleRandomGenerator),
            self._generate_feature("local_phase_distortion", H1LocalPhaseDistortionRandomGenerator),
            self._generate_feature("group_delay_ripple", H1GroupDelayRippleRandomGenerator),
            self._generate_feature("phase_noise", H1PhaseNoiseRandomGenerator),
        ]

        for feature in features:
            feature.save_csv(data_dir / f"{feature.name}.csv")

        combined = self._combine(features, name=f"{run_name}_combined")
        combined = self._enforce_physical_phase(combined, name=f"{run_name}_combined")
        combined.save_csv(data_dir / f"{combined.name}.csv")

        group_delay_ns = _group_delay_ns(combined.freq_hz, combined.phase_rad)
        return PhaseCombinedH1Run(
            run_name=run_name,
            data_dir=data_dir,
            results_dir=results_dir,
            single_features=features,
            combined=combined,
            group_delay_mean_ns=float(np.mean(group_delay_ns)),
            group_delay_positive_ratio=float(np.mean(group_delay_ns > 0.0)),
        )

    def _generate_feature(self, feature_name: str, generator_type: type) -> H1:
        generator_seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        generated = generator_type(seed=generator_seed).generate(name=feature_name)
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
        return H1(name=name, freq_hz=combined.freq_hz, h_db=combined.h_db, phase_rad=combined.phase_rad)

    def _enforce_physical_phase(self, h1: H1, name: str) -> H1:
        cfg = self.physical_constraints
        return _enforce_positive_group_delay_phase(
            h1,
            name=name,
            min_mean_group_delay_ns=cfg.min_mean_group_delay_ns,
            min_low_percentile_group_delay_ns=cfg.min_low_percentile_group_delay_ns,
            low_percentile=cfg.low_percentile,
        )


def _group_delay_ns(freq_hz: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    omega_rad = 2.0 * np.pi * freq_hz
    return -np.gradient(np.unwrap(phase_rad), omega_rad) * 1e9


def _enforce_positive_group_delay_phase(
    h1: H1,
    name: str,
    min_mean_group_delay_ns: float,
    min_low_percentile_group_delay_ns: float,
    low_percentile: float,
) -> H1:
    group_delay_ns = _group_delay_ns(h1.freq_hz, h1.phase_rad)
    mean_delay_ns = float(np.mean(group_delay_ns))
    low_delay_ns = float(np.percentile(group_delay_ns, low_percentile))
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


def plot_run(run: PhaseCombinedH1Run) -> list[Path]:
    plotter = HPhasePlotter(results_dir=run.results_dir)
    csv_files = [run.data_dir / f"{feature.name}.csv" for feature in run.single_features]
    csv_files.append(run.data_dir / f"{run.combined.name}.csv")
    return [plotter.plot_csv(csv_path) for csv_path in csv_files]


if __name__ == "__main__":
    generator = H1PhaseCombinedRandomGenerator()
    run = generator.generate()
    plot_paths = plot_run(run)

    print(f"run_name: {run.run_name}")
    print(f"data_folder: {run.data_dir}")
    print(f"results_folder: {run.results_dir}")
    print("single_features:")
    for feature in run.single_features:
        print(
            f"  {feature.name}: "
            f"phase_min_rad={np.min(feature.phase_rad):.6f}, "
            f"phase_max_rad={np.max(feature.phase_rad):.6f}"
        )
    print(
        f"combined: "
        f"phase_min_rad={np.min(run.combined.phase_rad):.6f}, "
        f"phase_max_rad={np.max(run.combined.phase_rad):.6f}"
    )
    print(f"group_delay_mean_ns: {run.group_delay_mean_ns:.6f}")
    print(f"group_delay_positive_ratio: {run.group_delay_positive_ratio:.6f}")
    print("saved_plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
