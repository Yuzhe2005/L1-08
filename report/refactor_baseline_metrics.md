# Refactor baseline metrics

Source run: `data/full_combined_20260610_203721` (seed 911204371)

| Metric | Value |
|---|---|
| H1 magnitude_combined_ripple_pp_db | 0.668567896 |
| L1-08 float dense ripple (ripple_after_db) | 0.109799118 |
| L1-08 fixed dense ripple (ripple_after_fixed_db) | 0.114183714 |
| Behavior ripple_after_fir_db | 0.083587240 |
| L1-08 QAM magnitude-only fixed EVM % | 0.232999286 |
| PlanB behavior ripple_after_plan_b_fixed_db | 0.038594987 |
| PlanB EVM_LIN fixed % | 0.355931417 |
| PlanB QAM fixed EVM % | 0.300096261 |

Post-refactor runs must match these values (same config seeds).
