import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


L1_09_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L1_09_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "results"
MPLCONFIG_ROOT = L1_09_ROOT / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GroupDelayInput:
    input_csv: Path
    run_name: str
    freq_hz: np.ndarray
    omega_rad: np.ndarray
    phase_rad: np.ndarray
    group_delay_ns: np.ndarray


@dataclass(frozen=True)
class AllPassDesign:
    input_data: GroupDelayInput
    output_dir: Path
    section_count: int
    target_delay_ns: float
    margin_ns: float
    smooth_window: int
    r_values: np.ndarray
    theta_values_rad: np.ndarray
    center_freq_hz: np.ndarray
    allpass_phase_rad: np.ndarray
    allpass_group_delay_ns: np.ndarray
    compensated_phase_rad: np.ndarray
    compensated_group_delay_ns: np.ndarray
    compensated_ripple_pp_ns: float
    compensated_rms_error_ns: float
    optimizer_cost: float
    optimizer_success: bool
    optimizer_message: str


def find_latest_group_delay_csv(results_root: Path = RESULTS_ROOT) -> Path:
    candidates = sorted(
        results_root.glob("h1_full_combined_random_*/l1_09_group_delay/group_delay_analysis.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No group_delay_analysis.csv found under {results_root}. "
            "Run L1_09_group_delay_analyzer.py first."
        )
    return candidates[0]


def load_group_delay_csv(input_csv: Path) -> GroupDelayInput:
    freq_hz: list[float] = []
    omega_rad: list[float] = []
    phase_rad: list[float] = []
    group_delay_ns: list[float] = []

    with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"freq_hz", "omega_rad", "phase_unwrapped_rad", "group_delay_ns"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_csv} must contain columns: {sorted(required_columns)}")

        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            omega_rad.append(float(row["omega_rad"]))
            phase_rad.append(float(row["phase_unwrapped_rad"]))
            group_delay_ns.append(float(row["group_delay_ns"]))

    freq = np.asarray(freq_hz, dtype=float)
    omega = np.asarray(omega_rad, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)
    delay = np.asarray(group_delay_ns, dtype=float)

    if freq.size < 16:
        raise ValueError("All-pass design needs at least 16 frequency points.")
    if not (freq.size == omega.size == phase.size == delay.size):
        raise ValueError("Input columns must have the same length.")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(omega)):
        raise ValueError("Frequency columns contain non-finite values.")
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(delay)):
        raise ValueError("Phase or group delay columns contain non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    run_name = input_csv.parents[1].name if input_csv.parent.name == "l1_09_group_delay" else input_csv.stem
    return GroupDelayInput(
        input_csv=input_csv,
        run_name=run_name,
        freq_hz=freq,
        omega_rad=omega,
        phase_rad=phase,
        group_delay_ns=delay,
    )


def default_output_dir(input_data: GroupDelayInput) -> Path:
    return RESULTS_ROOT / input_data.run_name / "l1_09_allpass_float"


def normalized_digital_frequency(freq_hz: np.ndarray) -> np.ndarray:
    span_hz = float(freq_hz[-1] - freq_hz[0])
    if span_hz <= 0.0:
        raise ValueError("Frequency span must be positive.")
    return np.pi * (freq_hz - freq_hz[0]) / span_hz


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def second_order_allpass_response(
    digital_w_rad: np.ndarray,
    r_values: np.ndarray,
    theta_values_rad: np.ndarray,
) -> np.ndarray:
    z_inv = np.exp(-1j * digital_w_rad)
    response = np.ones_like(z_inv, dtype=complex)

    for r, theta in zip(r_values, theta_values_rad):
        c = np.cos(theta)
        numerator = (r * r) - (2.0 * r * c * z_inv) + (z_inv * z_inv)
        denominator = 1.0 - (2.0 * r * c * z_inv) + ((r * r) * z_inv * z_inv)
        response *= numerator / denominator

    return response


def response_phase_and_delay_ns(
    digital_w_rad: np.ndarray,
    omega_rad: np.ndarray,
    r_values: np.ndarray,
    theta_values_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    response = second_order_allpass_response(digital_w_rad, r_values, theta_values_rad)
    phase_rad = np.unwrap(np.angle(response))
    delay_s = -np.gradient(phase_rad, omega_rad)
    return phase_rad, delay_s * 1e9


def pack_params(r_values: np.ndarray, theta_values_rad: np.ndarray) -> np.ndarray:
    return np.concatenate([r_values, theta_values_rad])


def unpack_params(params: np.ndarray, section_count: int) -> tuple[np.ndarray, np.ndarray]:
    r_values = params[:section_count]
    theta_values_rad = params[section_count:]
    return r_values, theta_values_rad


def initial_params(section_count: int) -> np.ndarray:
    r_values = np.full(section_count, 0.55, dtype=float)
    theta_values = np.linspace(0.12 * np.pi, 0.88 * np.pi, section_count)
    return pack_params(r_values, theta_values)


def parameter_bounds(section_count: int) -> tuple[np.ndarray, np.ndarray]:
    r_low = np.full(section_count, 0.05, dtype=float)
    r_high = np.full(section_count, 0.98, dtype=float)
    theta_low = np.full(section_count, 0.02 * np.pi, dtype=float)
    theta_high = np.full(section_count, 0.98 * np.pi, dtype=float)
    return pack_params(r_low, theta_low), pack_params(r_high, theta_high)


def design_allpass(
    input_data: GroupDelayInput,
    output_dir: Path,
    section_count: int,
    margin_ns: float | None,
    smooth_window: int,
) -> AllPassDesign:
    if section_count < 1:
        raise ValueError("section_count must be at least 1.")

    fit_delay_ns = moving_average(input_data.group_delay_ns, smooth_window)
    original_ripple_ns = float(np.max(fit_delay_ns) - np.min(fit_delay_ns))
    resolved_margin_ns = margin_ns
    if resolved_margin_ns is None:
        resolved_margin_ns = max(0.05, 0.05 * original_ripple_ns)

    target_delay_ns = float(np.max(fit_delay_ns) + resolved_margin_ns)
    digital_w = normalized_digital_frequency(input_data.freq_hz)

    initial_r_values, initial_theta_values = unpack_params(initial_params(section_count), section_count)
    _, initial_ap_delay_ns = response_phase_and_delay_ns(
        digital_w,
        input_data.omega_rad,
        initial_r_values,
        initial_theta_values,
    )
    initial_target_delay_ns = max(target_delay_ns, float(np.mean(fit_delay_ns + initial_ap_delay_ns)))
    target_lower_ns = target_delay_ns
    target_upper_ns = target_lower_ns + max(10.0, 3.0 * float(section_count))

    def residual(params: np.ndarray) -> np.ndarray:
        r_values, theta_values = unpack_params(params[:-1], section_count)
        candidate_target_delay_ns = params[-1]
        _, ap_delay_ns = response_phase_and_delay_ns(
            digital_w,
            input_data.omega_rad,
            r_values,
            theta_values,
        )
        compensated_delay_ns = fit_delay_ns + ap_delay_ns
        # Scale keeps the optimizer numerically comfortable while preserving ns units.
        return (compensated_delay_ns - candidate_target_delay_ns) / max(original_ripple_ns, 1.0)

    lower, upper = parameter_bounds(section_count)
    lower = np.concatenate([lower, np.asarray([target_lower_ns])])
    upper = np.concatenate([upper, np.asarray([target_upper_ns])])
    x0 = np.concatenate([initial_params(section_count), np.asarray([initial_target_delay_ns])])
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=2500,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    r_values, theta_values = unpack_params(result.x[:-1], section_count)
    target_delay_ns = float(result.x[-1])
    order = np.argsort(theta_values)
    r_values = r_values[order]
    theta_values = theta_values[order]

    allpass_phase, allpass_delay_ns = response_phase_and_delay_ns(
        digital_w,
        input_data.omega_rad,
        r_values,
        theta_values,
    )
    compensated_group_delay_ns = input_data.group_delay_ns + allpass_delay_ns
    compensated_phase_rad = input_data.phase_rad + allpass_phase
    center_freq_hz = input_data.freq_hz[0] + (theta_values / np.pi) * (
        input_data.freq_hz[-1] - input_data.freq_hz[0]
    )

    compensated_error = compensated_group_delay_ns - target_delay_ns
    compensated_ripple_pp_ns = float(np.max(compensated_group_delay_ns) - np.min(compensated_group_delay_ns))
    compensated_rms_error_ns = float(np.sqrt(np.mean(compensated_error**2)))

    return AllPassDesign(
        input_data=input_data,
        output_dir=output_dir,
        section_count=section_count,
        target_delay_ns=target_delay_ns,
        margin_ns=float(resolved_margin_ns),
        smooth_window=smooth_window,
        r_values=r_values,
        theta_values_rad=theta_values,
        center_freq_hz=center_freq_hz,
        allpass_phase_rad=allpass_phase,
        allpass_group_delay_ns=allpass_delay_ns,
        compensated_phase_rad=compensated_phase_rad,
        compensated_group_delay_ns=compensated_group_delay_ns,
        compensated_ripple_pp_ns=compensated_ripple_pp_ns,
        compensated_rms_error_ns=compensated_rms_error_ns,
        optimizer_cost=float(result.cost),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def save_coefficients_csv(design: AllPassDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "section",
                "r",
                "theta_rad",
                "center_freq_hz",
                "b0",
                "b1",
                "b2",
                "a0",
                "a1",
                "a2",
            ]
        )
        for idx, (r, theta, center_freq) in enumerate(
            zip(design.r_values, design.theta_values_rad, design.center_freq_hz),
            start=1,
        ):
            c = np.cos(theta)
            b0 = r * r
            b1 = -2.0 * r * c
            b2 = 1.0
            a0 = 1.0
            a1 = -2.0 * r * c
            a2 = r * r
            writer.writerow(
                [
                    idx,
                    f"{r:.12f}",
                    f"{theta:.12f}",
                    f"{center_freq:.6f}",
                    f"{b0:.12f}",
                    f"{b1:.12f}",
                    f"{b2:.12f}",
                    f"{a0:.12f}",
                    f"{a1:.12f}",
                    f"{a2:.12f}",
                ]
            )


def save_response_csv(design: AllPassDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = design.input_data
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "original_phase_rad",
                "allpass_phase_rad",
                "compensated_phase_rad",
                "original_group_delay_ns",
                "allpass_group_delay_ns",
                "compensated_group_delay_ns",
                "target_delay_ns",
            ]
        )
        for values in zip(
            data.freq_hz,
            data.phase_rad,
            design.allpass_phase_rad,
            design.compensated_phase_rad,
            data.group_delay_ns,
            design.allpass_group_delay_ns,
            design.compensated_group_delay_ns,
        ):
            (
                freq_hz,
                original_phase,
                allpass_phase,
                compensated_phase,
                original_delay,
                allpass_delay,
                compensated_delay,
            ) = values
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{original_phase:.12f}",
                    f"{allpass_phase:.12f}",
                    f"{compensated_phase:.12f}",
                    f"{original_delay:.9f}",
                    f"{allpass_delay:.9f}",
                    f"{compensated_delay:.9f}",
                    f"{design.target_delay_ns:.9f}",
                ]
            )


def save_metrics_csv(design: AllPassDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    original_delay = design.input_data.group_delay_ns
    original_mean = float(np.mean(original_delay))
    original_ripple = float(np.max(original_delay) - np.min(original_delay))
    original_rms = float(np.sqrt(np.mean((original_delay - original_mean) ** 2)))

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["input_csv", str(design.input_data.input_csv)])
        writer.writerow(["section_count", design.section_count])
        writer.writerow(["target_delay_ns", f"{design.target_delay_ns:.9f}"])
        writer.writerow(["target_margin_ns", f"{design.margin_ns:.9f}"])
        writer.writerow(["smooth_window", design.smooth_window])
        writer.writerow(["original_group_delay_ripple_pp_ns", f"{original_ripple:.9f}"])
        writer.writerow(["original_group_delay_rms_around_mean_ns", f"{original_rms:.9f}"])
        writer.writerow(["compensated_group_delay_ripple_pp_ns", f"{design.compensated_ripple_pp_ns:.9f}"])
        writer.writerow(["compensated_group_delay_rms_to_target_ns", f"{design.compensated_rms_error_ns:.9f}"])
        writer.writerow(["optimizer_cost", f"{design.optimizer_cost:.12e}"])
        writer.writerow(["optimizer_success", design.optimizer_success])
        writer.writerow(["optimizer_message", design.optimizer_message])


def plot_group_delay_before_after(design: AllPassDesign, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq = design.input_data.freq_hz
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freq, design.input_data.group_delay_ns, linewidth=1.3, label="before L1-09")
    ax.plot(freq, design.compensated_group_delay_ns, linewidth=1.3, label="after all-pass")
    ax.axhline(design.target_delay_ns, color="black", linestyle="--", linewidth=1.0, label="target")
    ax.set_title("L1-09 floating all-pass group delay")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_phase_before_after(design: AllPassDesign, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq = design.input_data.freq_hz
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freq, design.input_data.phase_rad, linewidth=1.3, label="before L1-09")
    ax.plot(freq, design.compensated_phase_rad, linewidth=1.3, label="after all-pass")
    ax.set_title("L1-09 floating all-pass phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design a floating multi-section second-order all-pass equalizer.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input group_delay_analysis.csv. Defaults to latest results/*/l1_09_group_delay/group_delay_analysis.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/<run_name>/l1_09_allpass_float.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=4,
        help="Number of second-order all-pass sections.",
    )
    parser.add_argument(
        "--margin-ns",
        type=float,
        default=None,
        help="Delay margin above max group delay. Defaults to max(0.05 ns, 5%% of smoothed ripple).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=31,
        help="Odd moving-average window used only for fitting the target delay shape.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or find_latest_group_delay_csv()
    input_data = load_group_delay_csv(input_csv)
    output_dir = args.output_dir or default_output_dir(input_data)

    design = design_allpass(
        input_data=input_data,
        output_dir=output_dir,
        section_count=args.sections,
        margin_ns=args.margin_ns,
        smooth_window=args.smooth_window,
    )

    coefficients_csv = output_dir / "allpass_coefficients.csv"
    response_csv = output_dir / "allpass_response.csv"
    metrics_csv = output_dir / "allpass_metrics.csv"
    group_delay_plot = output_dir / "group_delay_before_after_l1_09.png"
    phase_plot = output_dir / "phase_before_after_l1_09.png"

    save_coefficients_csv(design, coefficients_csv)
    save_response_csv(design, response_csv)
    save_metrics_csv(design, metrics_csv)
    plot_group_delay_before_after(design, group_delay_plot)
    plot_phase_before_after(design, phase_plot)

    print(f"input_csv: {input_csv}")
    print(f"output_dir: {output_dir}")
    print(f"section_count: {design.section_count}")
    print(f"target_delay_ns: {design.target_delay_ns:.9f}")
    print(f"compensated_group_delay_ripple_pp_ns: {design.compensated_ripple_pp_ns:.9f}")
    print(f"compensated_group_delay_rms_to_target_ns: {design.compensated_rms_error_ns:.9f}")
    print(f"optimizer_success: {design.optimizer_success}")
    print(f"coefficients_csv: {coefficients_csv}")
    print(f"response_csv: {response_csv}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"group_delay_plot: {group_delay_plot}")
    print(f"phase_plot: {phase_plot}")


if __name__ == "__main__":
    main()
