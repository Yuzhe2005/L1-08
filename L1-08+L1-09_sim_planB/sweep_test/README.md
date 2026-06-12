# Plan B Sweep Test

Sweep Plan B complex FIR design parameters with QAM EVM and EVM_LIN validation on a fixed H1 run.

## Config

Sweep parameters live in the repo-root file:

```text
config_plan_b_sweep.json
```

H1 / behavior / QAM input comes from `config_input.json` (same as Base Plan). Pipeline design defaults (non-sweep) remain in `config_plan_b.json`.

When `input.run_dir` is `null` in both `config_plan_b_sweep.json` and `config_plan_b.json`, the sweep generates a fresh H1 run before sweeping design parameters.

All sweep scripts run with **zero CLI arguments**. Edit `config_plan_b_sweep.json` only.

| Section | Purpose |
|---|---|
| `input` | optional fixed `run_dir` / `h1_csv` |
| `output` | sweep folder, `save_case_graphs`, `save_iq` |
| `design_sweep` | tap, regularization, fs_hz grids |
| `fixed_point_sweep` | quantization choices |
| `analysis` | analyzer targets and profiler `top_n` |

## Run

```powershell
python L1-08+L1-09_sim_planB\sweep_test\run_plan_b_sweep.py
```

Output goes to:

```text
sweep_result/<sweep_folder_name>/
```

Default folder name from `config_plan_b_sweep.json` is `plan_b_sweep_bw1g_3seed`:

```text
sweep_result/plan_b_sweep_bw1g_3seed/
```

## Analyze

```powershell
python L1-08+L1-09_sim_planB\sweep_test\analyze_plan_b_sweep_results.py
python L1-08+L1-09_sim_planB\sweep_test\sweep_result_profiler.py
```
