import os
import sys
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SIM_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
MPLCONFIG_ROOT = SIM_ROOT / ".matplotlib"


def ensure_repo_imports() -> None:
    for path in (REPO_ROOT, SIM_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


ensure_repo_imports()
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))
