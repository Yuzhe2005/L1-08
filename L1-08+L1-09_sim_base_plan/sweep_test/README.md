# Base Plan Sweep Test

Parameter sweep wrapper for the full L1-08 + L1-09 Base Plan pipeline.

## Config

Sweep parameters live in the repo-root file:

```text
config_base_plan_sweep.json
```

Runtime defaults and profiles come from `config_input.json` and `config_base_plan.json`. Seeds are injected per combo via environment variables (`L1_08_PROFILE`, `L1_08_H1_SEED`, etc.).

All sweep scripts run with **zero CLI arguments**. Edit `config_base_plan_sweep.json` only.

| Section | Purpose |
|---|---|
| `run.dry_run` | `true` = print combos only, no simulation |
| `output` | sweep folder name, cleanup, overwrite |
| `analysis` | analyzer ripple target (`target_ripple_db`) |
| `sweep` | profiles, seeds, tap/regularization/fixed-point grids |

## Run

```powershell
python L1-08+L1-09_sim_base_plan\sweep_test\run_sweep.py
```

Set `"dry_run": true` under `run` in the config to preview combos without executing.

Output goes to:

```text
sweep_result/<group_label>/<combo_folder>/
```

When `cleanup_sim_outputs_after_copy` is enabled in `config_base_plan_sweep.json`, temporary run folders under `data/` and `graph/` are removed after each combo is copied.

## Analyze

```powershell
python L1-08+L1-09_sim_base_plan\sweep_test\analyze_sweep_results.py
```
