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
class PhaseCombinedH1Run:
    run_name: str
    data_dir: Path
    results_dir: Path
    single_features: list[H1]
    combined: H1


class H1PhaseCombinedRandomGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

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

        combined = features[0]
        for feature in features[1:]:
            combined = combined.add(feature, name=f"{run_name}_combined")
        combined.save_csv(data_dir / f"{combined.name}.csv")

        return PhaseCombinedH1Run(
            run_name=run_name,
            data_dir=data_dir,
            results_dir=results_dir,
            single_features=features,
            combined=combined,
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
    print("saved_plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
