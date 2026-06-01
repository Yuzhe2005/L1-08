import csv
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class HPhaseData:
    name: str
    freq_hz: np.ndarray
    phase_rad: np.ndarray


class HPhasePlotter:
    def __init__(self, data_dir: Path | None = None, results_dir: Path | None = None) -> None:
        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.results_dir = results_dir or PROJECT_ROOT / "results"

    def load_csv(self, csv_path: Path) -> HPhaseData:
        freq_hz: list[float] = []
        phase_rad: list[float] = []

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"freq_hz", "phase_rad"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain columns: freq_hz,phase_rad")
            for row in reader:
                freq_hz.append(float(row["freq_hz"]))
                phase_rad.append(float(row["phase_rad"]))

        phase = np.unwrap(np.array(phase_rad))
        return HPhaseData(name=csv_path.stem, freq_hz=np.array(freq_hz), phase_rad=phase)

    def plot(self, data: HPhaseData, output_path: Path | None = None) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path or self.results_dir / f"{data.name}_phase.png"

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(data.freq_hz, data.phase_rad, linewidth=1.8)
        ax.set_title(f"{data.name} phase")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase (rad)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def plot_csv(self, csv_path: Path) -> Path:
        return self.plot(self.load_csv(csv_path))


if __name__ == "__main__":
    plotter = HPhasePlotter()
    csv_files = sorted(plotter.data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {plotter.data_dir}")
    print(f"saved_plot: {plotter.plot_csv(csv_files[-1])}")
