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
- 当前 Plan B 只完成 frequency-domain validation，还没有加入 complex I/Q input 或 QAM/EVM validation。

因此二者目前不是完全同一层级的验证：Base Plan 已经包含 time-domain QAM validation；Plan B 目前主要验证 complex FIR 在频域上是否可行。

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

Plan B 当前还没有做这一步。当前 Plan B 只是读取 H1 frequency response，并在 frequency domain 设计 complex FIR；它还没有模拟：

- complex I/Q input waveform；
- QAM symbols；
- H1 channel 对 QAM waveform 的影响；
- Plan B complex FIR 对 time-domain signal 的 filtering；
- fixed-point Plan B filtering；
- final EVM / constellation / per-bin EVM。

所以现在不能直接说 Plan B 在通信链路性能上已经优于 Base Plan，只能说 Plan B 在 frequency-domain compensation 上表现有潜力。

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

Plan B 的不足：

- 当前还没有 complex I/Q 或 QAM input simulation。
- 当前没有 EVM / constellation validation，因此不能直接和 Base Plan 的最终 EVM 做公平比较。
- 达到最佳 frequency-domain result 需要 320-tap complex FIR，资源代价高于 Base Plan。
- group delay 指标没有明显优于 Base Plan 的 all-pass IIR stage。

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

Plan B 下一步应该补齐 time-domain validation：

1. 生成和 Base Plan 一致的 QAM input / complex I/Q input。
2. 用 H1 frequency response 或等效 impulse response 作用到 input waveform。
3. 用 Plan B complex FIR 对 H1 output 做 filtering。
4. 加入 fixed-point coefficient 和 fixed-point filtering。
5. 输出 EVM summary、constellation graph、per-bin EVM。
6. 将 Plan B final EVM 与 Base Plan final EVM 直接比较。

只有完成这一步之后，才能正式判断 Plan B 是否真正优于 Base Plan。当前阶段的合理表述是：

> Plan B 在 frequency-domain compensation 上显示出较强潜力，尤其 320-tap Q18.15 complex FIR 可以取得更低的 magnitude ripple；但它还缺少 QAM/EVM chain validation，因此目前还不能作为完整替代 Base Plan 的结论。
