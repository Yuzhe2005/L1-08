# Plan B 与 Base Plan 对比分析

本文基于当前 baseline run `full_combined_20260605_090952` 以及 Plan B sweep test `h1_911204371_planB_behavior_1681887853_qam_825776271` 的结果，对 Base Plan 和 Plan B 做阶段性比较。

## 1. 两个方案的设计思路

Base Plan 采用两级补偿结构：

- L1-08：使用 64-tap real FIR 做 magnitude compensation。
- L1-09：使用 all-pass IIR 做 phase / group delay compensation。
- 后续包含 fixed-point quantization 和 QAM/EVM validation。

Plan B 采用单级补偿结构：

- 使用一个 complex FIR 同时补偿 H1 的 magnitude 和 phase。
- 当前 sweep 覆盖 tap number、regularization、fixed-point coefficient format。
- 当前已补齐 baseline QAM/EVM validation：`plan_b_qam_evm_validator.py` 使用和 Base Plan 一致的 QAM input、H1 channel、delay/gain fitting 与 EVM 计算方法，对 Plan B float/fixed complex FIR 做 time-domain filtering 验证。

因此现在二者已经可以在同一个 baseline run 上做 QAM/EVM 直接对比。但 Plan B 目前仍只完成单个 baseline run 的 QAM 验证，尚未像 Base Plan/L1-08 sweep 那样完成跨 bandwidth、seed、tap、fixed-point format 的大规模鲁棒性验证。

## 2. Frequency-Domain 补偿效果

Baseline H1 原始 magnitude ripple：

| 指标 | 数值 |
| --- | ---: |
| H1 magnitude ripple | 0.6686 dB |

Base Plan 的 L1-08 64-tap real FIR magnitude compensation：

| 指标 | 数值 |
| --- | ---: |
| tap number | 64 |
| regularization | 1e-4 |
| compensation 后 magnitude ripple | 0.1098 dB |
| fixed-point format | Q16.13 |
| fixed-point saturation | 0 |

Plan B sweep 中最推荐的结果：

| 指标 | 数值 |
| --- | ---: |
| tap number | 320 |
| regularization | 1e-5 |
| fixed-point format | Q18.15 |
| fixed magnitude ripple | 0.0598 dB |
| fixed phase RMS error | 0.00316 rad |
| fixed group delay ripple | 2.197 ns |
| estimated real multipliers | 1280 |
| fixed-point saturation | 0 |

从 frequency-domain magnitude compensation 看，Plan B 的最佳结果比 Base Plan 的 L1-08 magnitude stage 更好：

- Base Plan L1-08 after FIR：约 0.1098 dB ripple。
- Plan B best fixed complex FIR：约 0.0598 dB ripple。

但这个提升是用更高阶数和更高资源换来的。Base Plan 的 FIR 是 64 tap real FIR；Plan B 最佳结果是 320 tap complex FIR。

## 3. Phase / Group Delay 对比

Base Plan 的 phase / group delay 主要由 L1-09 all-pass IIR stage 处理。当前 fixed-point all-pass 结果：

| 指标 | 数值 |
| --- | ---: |
| all-pass section count | 8 |
| fixed-point format | Q18.15 |
| fixed-point saturation | 0 |
| fixed compensated group-delay ripple | 2.109 ns |
| fixed vs float phase RMS error | 3.80e-4 rad |
| stable | True |

Plan B 最佳结果：

| 指标 | 数值 |
| --- | ---: |
| fixed group-delay ripple | 2.197 ns |
| fixed phase RMS error vs reference delay | 0.00316 rad |

从 group delay ripple 看，Base Plan 的 L1-09 all-pass result 和 Plan B best result 在同一数量级：

- Base Plan fixed all-pass compensated group-delay ripple：约 2.109 ns。
- Plan B best fixed complex FIR group-delay ripple：约 2.197 ns。

这说明 Plan B 的单级 complex FIR 可以同时做 magnitude 和 phase compensation，但在 group delay 指标上并没有明显优于 Base Plan 的 all-pass stage。

## 4. Fixed-Point 结果

Base Plan：

- L1-08 FIR 使用 Q16.13，saturation count = 0。
- L1-09 all-pass 使用 Q18.15，saturation count = 0。
- fixed-point 后 all-pass 仍稳定。

Plan B sweep：

- 最佳结果使用 Q18.15，saturation count = 0。
- Q18.15 整体比 Q16.13 更稳，尤其在 256 tap / 320 tap 区间。
- 一些低 regularization 的低 tap case 出现明显 coefficient saturation，例如 128 tap / 1e-6 和 192 tap / 1e-6，会导致 magnitude、phase、group delay 指标严重恶化。

因此 Plan B 如果继续推进，当前推荐 default fixed-point format 是 Q18.15。

## 5. QAM / EVM Validation 覆盖程度

Base Plan 已经完成 QAM/EVM validation：

| stage | EVM | magnitude-only EVM |
| --- | ---: | ---: |
| after H1 | 11.733% | 1.633% |
| after L1-08 fixed FIR | 11.551% | 0.233% |
| after L1-08 fixed FIR + L1-09 fixed all-pass | 3.397% | 2.263% |

这个结果说明：

- L1-08 主要改善 magnitude distortion。
- L1-09 all-pass 主要改善 phase / delay distortion。
- 最终 EVM 从 11.733% 降到 3.397%，说明 Base Plan 的完整链路验证是有效的。

Plan B 现在已经补齐 baseline QAM/EVM validation。新增脚本：

```powershell
python L1-08+L1-09_sim_planB\plan_b_qam_evm_validator.py --run-dir data\full_combined_20260605_090952
```

该脚本完成：

- 生成和 Base Plan 一致的 QAM-loaded IF input。
- 使用同一个 H1 frequency response 作用到 QAM waveform。
- 对 H1 output 分别应用 Plan B float complex FIR 和 fixed complex FIR。
- 输出 `plan_b_qam_evm_summary.csv`、`plan_b_qam_per_bin.csv` 和 `plan_b_qam_evm.png`。

当前已进一步完成一个小规模 Plan B QAM sweep，覆盖：

```text
tap_num: 256, 320
regularization: 1e-6, 1e-5
fixed-point format: Q18.15
run: full_combined_20260605_090952
```

输出位于：

```text
sweep_result/plan_b_sweep_bw1g_3seed/
```

小 sweep 的 QAM/EVM 结果如下：

| Plan B case | fixed EVM | fixed magnitude-only EVM | fixed magnitude ripple | multipliers |
| --- | ---: | ---: | ---: | ---: |
| 256 tap, 1e-6, Q18.15 | 0.300% | 0.124% | 0.0750 dB | 1024 |
| 256 tap, 1e-5, Q18.15 | 0.306% | 0.131% | 0.0742 dB | 1024 |
| 320 tap, 1e-6, Q18.15 | 0.274% | 0.0916% | 0.0610 dB | 1280 |
| 320 tap, 1e-5, Q18.15 | 0.274% | 0.0924% | 0.0598 dB | 1280 |

从这个 baseline run 看，Plan B fixed complex FIR 的最终 QAM EVM 明显低于 Base Plan fixed all-pass 结果（best 0.274% vs 3.397%）。其中 320 tap 的性能最好，但需要约 1280 个 real multipliers；256 tap 的 EVM 也在 0.30% 左右，资源为约 1024 个 real multipliers。这个结果说明 Plan B 是很强的候选方案，但仍需要跨 seed / bandwidth 做鲁棒性验证，不能只凭一个 run 作为最终架构选择依据。

## 6. Resource / Complexity 对比

Base Plan：

- L1-08：64-tap real FIR。
- L1-09：8-section all-pass IIR。
- 结构分工明确，magnitude 和 phase 分开处理。
- 已经有完整 QAM/EVM validation。

Plan B：

- 最佳结果为 320-tap complex FIR。
- complex FIR 每个 tap 包含 complex coefficient，估算 real multiplier 数量为 1280。
- 结构上更统一，只需要一个 module 处理 magnitude + phase。
- 但硬件资源代价明显更高。

如果只看 frequency-domain 指标，Plan B best case 很强；如果看资源效率，256 tap Plan B 可能是更合理的折中：

| Plan B case | fixed magnitude ripple | fixed phase RMS error | fixed group-delay ripple | multipliers |
| --- | ---: | ---: | ---: | ---: |
| 320 tap, 1e-5, Q18.15 | 0.0598 dB | 0.00316 rad | 2.197 ns | 1280 |
| 256 tap, 1e-6, Q18.15 | 0.0748 dB | 0.00328 rad | 2.096 ns | 1024 |
| 256 tap, 1e-5, Q18.15 | 0.0742 dB | 0.00330 rad | 2.144 ns | 1024 |

因此，如果导师更关注 ultimate compensation performance，可以优先看 320 tap；如果关注 resource-performance trade-off，256 tap 更值得继续验证。

## 7. 当前结论

Plan B 的优点：

- 单个 complex FIR 可以同时补偿 magnitude 和 phase，结构概念上更直接。
- Sweep 结果显示 320 tap / 1e-5 / Q18.15 可以把 fixed magnitude ripple 降到约 0.0598 dB。
- 256 tap / Q18.15 也可以达到约 0.074 dB ripple，是比较好的资源折中点。
- fixed-point Q18.15 在最佳区域没有 saturation，量化误差较小。
- baseline QAM sweep 已补齐；当前 4 个关键 case 均无 saturation，fixed Plan B 可将 full EVM 从 11.733% 降到约 0.274%~0.306%。

Plan B 的不足：

- 达到最佳 frequency-domain result 需要 320-tap complex FIR，资源代价高于 Base Plan。
- group delay 指标没有明显优于 Base Plan 的 all-pass IIR stage。
- 目前 QAM sweep 只覆盖 baseline run，尚未完成跨 seed / bandwidth / fixed-point format 的鲁棒性验证。

Base Plan 的优点：

- 已经完成从 H1 generation、magnitude compensation、phase compensation、fixed-point 到 QAM/EVM 的完整 pipeline。
- 最终 fixed-point QAM EVM 从 11.733% 降到 3.397%。
- 结构模块化，L1-08 和 L1-09 分工清楚。
- 资源上比 320-tap complex FIR 更容易解释和实现。

Base Plan 的不足：

- 需要两个 compensation modules。
- L1-08 magnitude compensation 后 ripple 约 0.1098 dB，弱于 Plan B best frequency-domain result。
- L1-09 all-pass 只处理 phase / group delay，不直接改善 magnitude。

## 8. 下一步建议

Plan B 的 baseline time-domain validation 和小规模 QAM sweep 已经补齐。下一步应该把验证范围扩展到跨 seed / bandwidth：

1. 在至少 3 组 seed 和多个 bandwidth profile 上重复 QAM/EVM 验证。
2. 重点保留 256 tap / 1e-6 / Q18.15 与 320 tap / 1e-6 / Q18.15 两个候选点。
3. 增加 Plan B 与 Base Plan 的统一 summary table，包括 final EVM、magnitude-only EVM、group-delay ripple、multiplier estimate、fixed-point saturation。
4. 如果 Plan B 在跨 seed / bandwidth 上仍稳定，再进一步评估 RTL 资源和延迟是否可以接受。

当前阶段的合理表述是：

> Plan B 在 frequency-domain compensation 和 baseline QAM/EVM sweep 上都显示出很强潜力。当前 320-tap Q18.15 fixed complex FIR 在 baseline run 中可将 final QAM EVM 降到约 0.274%，256-tap Q18.15 fixed complex FIR 也可达到约 0.300%，均优于 Base Plan 当前 3.397% 的 fixed all-pass 结果。但 Plan B 的资源代价明显更高，且仍缺少跨 seed / bandwidth 的鲁棒性验证，因此现阶段应作为强候选方案继续扩展验证，而不是立即替代 Base Plan 进入 RTL。
