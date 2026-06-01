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
class PhaseNoiseRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    noise_std_rad_min: float = 0.001
    noise_std_rad_max: float = 0.01


class H1PhaseNoiseRandomGenerator:
    def __init__(self, config: PhaseNoiseRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or PhaseNoiseRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_phase_noise_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()

        noise_std_rad = self.rng.uniform(cfg.noise_std_rad_min, cfg.noise_std_rad_max)
        phase_rad = self.rng.normal(loc=0.0, scale=noise_std_rad, size=freq_hz.size)
        phase_rad -= np.mean(phase_rad)
        phase_rad = np.unwrap(phase_rad)
        h_db = np.zeros_like(freq_hz, dtype=float)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db, phase_rad=phase_rad)


if __name__ == "__main__":
    generator = H1PhaseNoiseRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_phase_noise_random_{timestamp}")

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
