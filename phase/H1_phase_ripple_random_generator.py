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
class PhaseRippleRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    component_count_min: int = 1
    component_count_max: int = 4
    amplitude_rad_min: float = 0.02
    amplitude_rad_max: float = 0.20
    cycles_min: float = 0.5
    cycles_max: float = 5.0


class H1PhaseRippleRandomGenerator:
    def __init__(self, config: PhaseRippleRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or PhaseRippleRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_phase_ripple_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()
        x = np.linspace(0.0, 1.0, freq_hz.size)

        component_count = int(self.rng.integers(cfg.component_count_min, cfg.component_count_max + 1))
        phase_rad = np.zeros_like(freq_hz, dtype=float)

        for _ in range(component_count):
            amplitude_rad = self.rng.uniform(cfg.amplitude_rad_min, cfg.amplitude_rad_max)
            cycles = self.rng.uniform(cfg.cycles_min, cfg.cycles_max)
            phase_offset_rad = self.rng.uniform(0.0, 2.0 * np.pi)
            phase_rad += amplitude_rad * np.sin(2.0 * np.pi * cycles * x + phase_offset_rad)

        phase_rad -= np.mean(phase_rad)
        phase_rad = np.unwrap(phase_rad)
        h_db = np.zeros_like(freq_hz, dtype=float)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db, phase_rad=phase_rad)


if __name__ == "__main__":
    generator = H1PhaseRippleRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_phase_ripple_random_{timestamp}")

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
