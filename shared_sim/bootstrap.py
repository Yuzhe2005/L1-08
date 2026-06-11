import os
import sys
from pathlib import Path

SHARED_SIM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SHARED_SIM_ROOT.parent
BASE_PLAN_ROOT = REPO_ROOT / "L1-08+L1-09_sim_base_plan"
L1_08_SIM_ROOT = BASE_PLAN_ROOT / "L1-08_sim"
L1_09_SIM_ROOT = BASE_PLAN_ROOT / "L1_09_sim"
PLAN_B_ROOT = REPO_ROOT / "L1-08+L1-09_sim_planB"
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "graph"
MPLCONFIG_ROOT = SHARED_SIM_ROOT / ".matplotlib"


def ensure_repo_imports() -> None:
    for path in (REPO_ROOT, L1_08_SIM_ROOT, L1_09_SIM_ROOT, SHARED_SIM_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


ensure_repo_imports()
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))
