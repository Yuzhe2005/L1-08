import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from L1_08_config import load_l1_08_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def update_run_summary(
    run_dir: Path,
    stage_name: str,
    stage_summary: dict[str, Any],
    graph_dir: Path | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    now = _utc_now()

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "run_name": run_dir.name,
            "data_dir": str(run_dir),
            "created_at_utc": now,
            "stages": {},
            "stage_history": [],
        }

    summary["updated_at_utc"] = now
    summary["data_dir"] = str(run_dir)
    if graph_dir is not None:
        summary["graph_dir"] = str(graph_dir)
    summary["experiment_config"] = _jsonable(load_l1_08_config())

    stages = summary.setdefault("stages", {})
    stages[stage_name] = _jsonable(stage_summary)
    summary.setdefault("stage_history", []).append(
        {
            "stage": stage_name,
            "updated_at_utc": now,
        }
    )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path
