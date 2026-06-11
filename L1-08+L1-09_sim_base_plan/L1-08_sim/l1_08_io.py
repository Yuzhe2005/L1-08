from pathlib import Path

from shared_sim.io_utils import (  # noqa: F401
    BASE_RUN_NAME_PREFIX,
    H1Magnitude,
    H1Phase,
    H1_DATA_DIR_NAME,
    PLAN_B_RUN_NAME_PREFIX,
    find_latest_h1_run,
    find_latest_ready_run,
    h1_data_dir,
    load_fir_coefficients,
    load_h1_magnitude,
    load_h1_phase,
    run_dir_from_data_path as _shared_run_dir_from_data_path,
    save_iq_csv,
)
from shared_sim.paths import DATA_ROOT, L1_08_SIM_ROOT

PROJECT_ROOT = SIM_ROOT = L1_08_SIM_ROOT
REPO_ROOT = PROJECT_ROOT.parent

L1_08_H2_TARGET_DATA_DIR_NAME = "l1_08_h2_target"
L1_08_H2_FIR_DESIGN_DATA_DIR_NAME = "l1_08_h2_fir_design"
L1_08_H2_FIXED_POINT_DATA_DIR_NAME = "l1_08_h2_fixed_point"
L1_08_BEHAVIOR_DATA_DIR_NAME = "l1_08_behavior"
L1_08_QAM_EVM_DATA_DIR_NAME = "l1_08_qam_evm"

STAGE_DATA_DIR_NAMES = {
    H1_DATA_DIR_NAME,
    L1_08_H2_TARGET_DATA_DIR_NAME,
    L1_08_H2_FIR_DESIGN_DATA_DIR_NAME,
    L1_08_H2_FIXED_POINT_DATA_DIR_NAME,
    L1_08_BEHAVIOR_DATA_DIR_NAME,
    L1_08_QAM_EVM_DATA_DIR_NAME,
}


def h2_target_data_dir(run_dir: Path) -> Path:
    return run_dir / L1_08_H2_TARGET_DATA_DIR_NAME


def h2_fir_design_data_dir(run_dir: Path) -> Path:
    return run_dir / L1_08_H2_FIR_DESIGN_DATA_DIR_NAME


def h2_fixed_point_data_dir(run_dir: Path) -> Path:
    return run_dir / L1_08_H2_FIXED_POINT_DATA_DIR_NAME


def behavior_data_dir(run_dir: Path) -> Path:
    return run_dir / L1_08_BEHAVIOR_DATA_DIR_NAME


def qam_evm_data_dir(run_dir: Path) -> Path:
    return run_dir / L1_08_QAM_EVM_DATA_DIR_NAME


def run_dir_from_data_path(path: Path) -> Path:
    return _shared_run_dir_from_data_path(path, STAGE_DATA_DIR_NAMES)
