from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "L1-08_sim"))
sys.path.insert(0, str(PROJECT_ROOT))

from H1_common import FrequencyGridConfig, H1


@dataclass(frozen=True)
class LocalPhaseDistortionRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    feature_count_min: int = 1
    feature_count_max: int = 4
    amplitude_rad_min: float = 0.03
    amplitude_rad_max: float = 0.30
    width_fraction_min: float = 0.03
    width_fraction_max: float = 0.15
    center_fraction_min: float = 0.05
    center_fraction_max: float = 0.95


class H1LocalPhaseDistortionRandomGenerator:
    def __init__(self, config: LocalPhaseDistortionRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or LocalPhaseDistortionRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_local_phase_distortion_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()
        x = np.linspace(0.0, 1.0, freq_hz.size)

        feature_count = int(self.rng.integers(cfg.feature_count_min, cfg.feature_count_max + 1))
        phase_rad = np.zeros_like(freq_hz, dtype=float)

        for _ in range(feature_count):
            amplitude_rad = self.rng.uniform(cfg.amplitude_rad_min, cfg.amplitude_rad_max)
            polarity = self.rng.choice([-1.0, 1.0])
            center = self.rng.uniform(cfg.center_fraction_min, cfg.center_fraction_max)
            width = self.rng.uniform(cfg.width_fraction_min, cfg.width_fraction_max)

            gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
            phase_rad += polarity * amplitude_rad * gaussian

        phase_rad -= np.mean(phase_rad)
        phase_rad = np.unwrap(phase_rad)
        h_db = np.zeros_like(freq_hz, dtype=float)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db, phase_rad=phase_rad)


if __name__ == "__main__":
    generator = H1LocalPhaseDistortionRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_local_phase_distortion_random_{timestamp}")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"magnitude_ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"phase_min_rad: {np.min(h1.phase_rad):.6f}")
    print(f"phase_max_rad: {np.max(h1.phase_rad):.6f}")
    print(f"saved_csv: {output_path}")
