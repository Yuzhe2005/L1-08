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
class HData:
    name: str
    freq_hz: np.ndarray
    h_db: np.ndarray

    def __post_init__(self) -> None:
        freq_hz = np.asarray(self.freq_hz, dtype=float)
        h_db = np.asarray(self.h_db, dtype=float)

        if freq_hz.ndim != 1:
            raise ValueError("freq_hz must be a 1-D array.")
        if h_db.ndim != 1:
            raise ValueError("h_db must be a 1-D array.")
        if freq_hz.size != h_db.size:
            raise ValueError("freq_hz and h_db must have the same length.")
        if freq_hz.size < 2:
            raise ValueError("H data needs at least two frequency points.")
        if not np.all(np.isfinite(freq_hz)):
            raise ValueError("freq_hz contains non-finite values.")
        if not np.all(np.isfinite(h_db)):
            raise ValueError("h_db contains non-finite values.")

        object.__setattr__(self, "freq_hz", freq_hz)
        object.__setattr__(self, "h_db", h_db)


class HPlotter:
    def __init__(self, data_dir: Path | None = None, graph_dir: Path | None = None) -> None:
        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.graph_dir = graph_dir or PROJECT_ROOT / "graph"

    def load_csv(self, csv_path: Path) -> HData:
        freq_hz: list[float] = []
        h_db: list[float] = []

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"freq_hz", "h_db"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain columns: freq_hz,h_db")

            for row in reader:
                freq_hz.append(float(row["freq_hz"]))
                h_db.append(float(row["h_db"]))

        return HData(name=csv_path.stem, freq_hz=np.array(freq_hz), h_db=np.array(h_db))

    def plot(self, h_data: HData, output_path: Path | None = None) -> Path:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path or self.graph_dir / f"{h_data.name}.png"

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(h_data.freq_hz, h_data.h_db, linewidth=1.8)
        ax.set_title(h_data.name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude Error (dB)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        return output_path

    def plot_csv(self, csv_path: Path) -> Path:
        h_data = self.load_csv(csv_path)
        return self.plot(h_data)

    def plot_latest_csv(self) -> Path:
        csv_files = sorted(self.data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        return self.plot_csv(csv_files[-1])


if __name__ == "__main__":
    plotter = HPlotter()
    output_path = plotter.plot_latest_csv()
    print(f"saved_plot: {output_path}")
