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
class LinearPhaseDelayRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    delay_ns_min: float = 1.0
    delay_ns_max: float = 4.0
    allow_negative_delay: bool = False


class H1LinearPhaseDelayRandomGenerator:
    def __init__(self, config: LinearPhaseDelayRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or LinearPhaseDelayRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_linear_phase_delay_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()

        delay_ns = self.rng.uniform(cfg.delay_ns_min, cfg.delay_ns_max)
        if cfg.allow_negative_delay:
            delay_ns *= self.rng.choice([-1.0, 1.0])

        delay_s = delay_ns * 1e-9
        h_db = np.zeros_like(freq_hz, dtype=float)
        phase_rad = -2.0 * np.pi * freq_hz * delay_s
        phase_rad = np.unwrap(phase_rad)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db, phase_rad=phase_rad)


if __name__ == "__main__":
    generator = H1LinearPhaseDelayRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_linear_phase_delay_random_{timestamp}")

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
