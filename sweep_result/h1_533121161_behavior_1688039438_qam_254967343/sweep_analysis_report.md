# L1-08 Sweep Analysis Report

## 1. Scope

This report summarizes one completed L1-08 parameter sweep from `sweep_summary.csv`.

- Total combos: `27`
- tap_num values: `64, 80, 96`
- regularization values: `0.0001, 0.0003, 0.001`
- fixed-point formats: `Q2.14, Q3.13, Q4.12`
- H1 ripple before compensation: `0.445925 dB`
- Ripple pass target used in this report: `0.100000 dB`

## 2. Overall Result

- Fixed dense ripple pass count: `15 / 27`
- Fixed multi-tone behavior pass count: `20 / 27`
- Saturated combo count: `0 / 27`

## 3. Best Combos

| Criterion | Combo | Dense ripple (dB) | Behavior ripple (dB) | QAM mag-only EVM (%) | Saturation |
|---|---|---:|---:|---:|---:|
| best_fixed_dense | `tap096_reg1em04_q2_14` | 0.028068 | 0.019036 | 0.042763 | 0 |
| best_fixed_dense_unsaturated | `tap096_reg1em04_q2_14` | 0.028068 | 0.019036 | 0.042763 | 0 |
| best_behavior_fixed | `tap096_reg1em04_q3_13` | 0.031837 | 0.018560 | 0.056873 | 0 |
| best_behavior_fixed_unsaturated | `tap096_reg1em04_q3_13` | 0.031837 | 0.018560 | 0.056873 | 0 |
| best_qam_fixed | `tap096_reg1em04_q2_14` | 0.028068 | 0.019036 | 0.042763 | 0 |
| best_qam_fixed_unsaturated | `tap096_reg1em04_q2_14` | 0.028068 | 0.019036 | 0.042763 | 0 |
| lowest_tap_dense_pass | `tap080_reg1em04_q2_14` | 0.054026 | 0.044133 | 0.140755 | 0 |
| lowest_tap_behavior_pass | `tap064_reg1em04_q2_14` | 0.127234 | 0.097746 | 0.351125 | 0 |

## 4. Group Summary

### By Tap

| Group | Combos | Dense pass | Behavior pass | Saturated | Best dense (dB) | Best behavior (dB) | Best QAM mag EVM (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9 | 0 | 2 | 0 | 0.124505 | 0.097746 | 0.351049 |
| 80 | 9 | 6 | 9 | 0 | 0.054026 | 0.044133 | 0.140755 |
| 96 | 9 | 9 | 9 | 0 | 0.028068 | 0.018560 | 0.042763 |

### By Regularization

| Group | Combos | Dense pass | Behavior pass | Saturated | Best dense (dB) | Best behavior (dB) | Best QAM mag EVM (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 9 | 6 | 8 | 0 | 0.028068 | 0.018560 | 0.042763 |
| 0.0003 | 9 | 6 | 6 | 0 | 0.030002 | 0.021159 | 0.047799 |
| 0.001 | 9 | 3 | 6 | 0 | 0.029002 | 0.018991 | 0.050668 |

### By Fixed-Point Format

| Group | Combos | Dense pass | Behavior pass | Saturated | Best dense (dB) | Best behavior (dB) | Best QAM mag EVM (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2.14 | 9 | 5 | 7 | 0 | 0.028068 | 0.018991 | 0.042763 |
| Q3.13 | 9 | 5 | 7 | 0 | 0.031837 | 0.018560 | 0.051070 |
| Q4.12 | 9 | 5 | 6 | 0 | 0.034711 | 0.024548 | 0.056229 |

## 5. Interpretation

- tap_num `64` did not pass dense `0.1 dB` in this sweep. It is not a robust choice for this H1 seed.
- tap_num `96` passed dense `0.1 dB` for every swept regularization/fixed-point format.
- Lowest-tap dense-pass candidate: `tap080_reg1em04_q2_14` with fixed dense ripple `0.054026 dB` and QAM magnitude-only EVM `0.140755%`.
- Best unsaturated QAM magnitude-only EVM candidate: `tap096_reg1em04_q2_14` with `0.042763%`.
- Dense ripple should be treated as the stricter pass/fail metric because multi-tone verification samples only selected frequencies and may miss the worst point in the full H1 grid.

## 6. Generated Files

- Best combo table: `sweep_best_combos.csv`
- Group summary table: `sweep_group_summary.csv`
- Plot: `sweep_fixed_dense_ripple_by_tap.png`
- Plot: `sweep_behavior_ripple_by_tap.png`
- Plot: `sweep_qam_evm_by_tap.png`
- Plot: `sweep_saturation_and_coeff_range.png`

## 7. Plots

![sweep_fixed_dense_ripple_by_tap](sweep_fixed_dense_ripple_by_tap.png)

![sweep_behavior_ripple_by_tap](sweep_behavior_ripple_by_tap.png)

![sweep_qam_evm_by_tap](sweep_qam_evm_by_tap.png)

![sweep_saturation_and_coeff_range](sweep_saturation_and_coeff_range.png)
