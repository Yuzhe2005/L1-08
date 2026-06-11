from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_sim.h1_common import FrequencyGridConfig, H1
from shared_sim.paths import REPO_ROOT as PROJECT_ROOT


@dataclass(frozen=True)
class RippleRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    component_count_min: int = 1
    component_count_max: int = 4
    amplitude_db_min: float = 0.02
    amplitude_db_max: float = 0.12
    cycles_min: float = 0.5
    cycles_max: float = 5.0


class H1RippleRandomGenerator:
    def __init__(self, config: RippleRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or RippleRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_ripple_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()
        x = np.linspace(0.0, 1.0, freq_hz.size)

        component_count = int(self.rng.integers(cfg.component_count_min, cfg.component_count_max + 1))
        h_db = np.zeros_like(freq_hz, dtype=float)

        for _ in range(component_count):
            amplitude_db = self.rng.uniform(cfg.amplitude_db_min, cfg.amplitude_db_max)
            cycles = self.rng.uniform(cfg.cycles_min, cfg.cycles_max)
            phase_rad = self.rng.uniform(0.0, 2.0 * np.pi)
            h_db += amplitude_db * np.sin(2.0 * np.pi * cycles * x + phase_rad)

        # Keep ripple centered around 0 dB so it represents shape, not gain offset.
        h_db -= np.mean(h_db)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db)


if __name__ == "__main__":
    generator = H1RippleRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_ripple_random_{timestamp}")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"saved_csv: {output_path}")
