import numpy as np
from scipy.signal import lfilter


def evaluate_fir_response(coeffs: np.ndarray, freq_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz / fs_hz
    n = np.arange(coeffs.size, dtype=float)
    return np.exp(-1j * np.outer(omega, n)) @ coeffs


def apply_fir_with_cyclic_prefix(block: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    cp_len = coeffs.size - 1
    if block.size <= cp_len:
        raise ValueError("Input block must be longer than FIR cyclic prefix length.")

    prefixed = np.concatenate([block[-cp_len:], block])
    filtered = lfilter(coeffs, [1.0], prefixed)
    return filtered[cp_len : cp_len + block.size]
