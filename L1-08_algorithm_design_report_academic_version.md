# L1-08 幅频 FIR 均衡算法行为级仿真报告（学术版）

## 摘要

本文针对 L1-08 幅频 FIR 均衡算法进行行为级仿真验证。L1-08 的目标是在射频前端链路存在幅频不平坦的情况下，通过数字 FIR 均衡器补偿通带内 magnitude ripple，使补偿后的总响应尽量平坦。本文构建了一个可复现的仿真流程，包括随机复数通道响应 H1 生成、理想补偿目标 H2_target 生成、real linear-phase FIR 设计、fixed-point 系数量化、multi-tone 行为级验证和 QAM magnitude-only EVM 辅助验证。

根据导师反馈，本文将仿真从单个随机 case 扩展为 bandwidth profile 与 seed case 两个维度的 sweep。当前实验共覆盖 5 个 bandwidth profile、3 组 seed case、3 个 FIR tap 数和 3 种 fixed-point 格式，共 135 组组合。实验结果显示，fixed dense ripple 通过 0.1 dB 指标的组合为 101/135，multi-tone behavior ripple 通过 0.1 dB 指标的组合为 122/135。结果说明 L1-08 的算法方向是可行的，但实际鲁棒性受 H1 随机形态、FIR tap 数和 fixed-point 格式影响明显，后续仍需继续进行跨 seed、跨 bandwidth 和跨 fixed-point 的鲁棒性验证。

---

## 1. 术语与缩略词

| 术语 | 含义 |
|---|---|
| FIR | Finite Impulse Response，有限冲激响应滤波器 |
| tap | FIR 系数个数 |
| H1 | 硬件前端链路的复数频率响应 |
| H2 | 数字 FIR 补偿滤波器的频率响应 |
| H2_target | 根据 H1 magnitude 构造的理想反向补偿目标 |
| Htotal | H1 与 H2 串联后的总响应 |
| magnitude ripple | 通带内幅度最大值与最小值之差 |
| dense ripple | 在完整频率网格上计算的 residual magnitude ripple |
| behavior ripple | 通过 multi-tone 时域仿真测得的 tone amplitude ripple |
| complex I/Q | 由 I + jQ 表示的复数信号 |
| IF | Intermediate Frequency，中频 |
| QAM | Quadrature Amplitude Modulation，正交幅度调制 |
| EVM | Error Vector Magnitude，误差向量幅度 |
| LS | Least Squares，最小二乘 |
| ridge regularization | 用于限制系数过大的正则化方法 |
| fixed-point | 固定位宽的定点数表示 |
| saturation | 定点数超出可表示范围后被截断到最大或最小值 |
| Qm.n | 定点格式，其中 m 为整数位数，n 为小数位数 |

---

## 2. 问题背景与目标

### 2.1 L1-08 需要解决的问题

射频前端链路中存在多种非理想器件，例如预选滤波器、cavity/LTCC、电缆、PCB 走线和不同频段下的链路差异。这些因素会导致通带内增益不完全平坦，从而产生 magnitude ripple。

在宽带测量或宽带调制场景中，magnitude ripple 会导致不同频率成分幅度不一致，进而影响系统指标。L1-08 的目标是通过数字 FIR 均衡器对幅频不平坦进行补偿，使补偿后的总 magnitude response 更接近平坦。

### 2.2 算法目标

本文当前验证的主要目标为：

1. 根据随机生成或实际测得的 H1 magnitude，构造理想反向补偿目标 H2_target。
2. 使用有限 tap 的 real linear-phase FIR 逼近 H2_target。
3. 验证 fixed-point 量化后，H1 与 H2_fixed 串联后的 residual magnitude ripple 是否可控制在 0.1 dB 以内。
4. 使用 multi-tone complex I/Q IF 信号进行行为级验证。
5. 使用 QAM magnitude-only EVM 作为辅助观察指标。

本文暂不以 phase/group delay correction 为主要目标。phase 相关随机特征目前用于增强 H1 模型的真实性，并为后续 L1-09 phase/group delay 修复预留接口。

---

## 3. 理论模型

### 3.1 H1 通道模型

H1 表示硬件前端链路的复数频率响应：

```text
H1(f) = |H1(f)| * exp(j * phase_H1(f))
```

其中，`|H1(f)|` 表示链路幅频响应，`phase_H1(f)` 表示链路相频响应。L1-08 当前主要关注 `|H1(f)|` 的不平坦问题。

### 3.2 H2 补偿目标

理想补偿目标可以理解为 H1 magnitude 的反向响应：

```text
H2_target(f) ≈ 1 / |H1(f)|
```

这样可以使：

```text
|H1(f)| * |H2_target(f)| ≈ constant
```

实际实现中，H2_target 经过归一化处理，避免整体增益被无意义放大或缩小。

### 3.3 FIR 实际实现

由于硬件中 FIR tap 数有限，实际 FIR 响应 H2_actual 无法完全等于 H2_target。当前程序使用 least-squares 方法设计 real linear-phase FIR，使 H2_actual 尽量逼近 H2_target。

最终评价对象是：

```text
Htotal(f) = H1(f) * H2_actual(f)
```

若 `|Htotal(f)|` 在通带内足够平坦，则说明补偿有效。

---

## 4. 行为级仿真方案设计

### 4.1 总体流程

当前行为级仿真 pipeline 如下：

```text
H1 random generation
        ↓
H2 target generation
        ↓
FIR coefficient design
        ↓
fixed-point quantization
        ↓
multi-tone behavior simulation
        ↓
QAM magnitude-only EVM simulation
        ↓
sweep result analysis
```

该流程先在行为级验证算法方向，再为后续 RTL 实现和真实硬件数据对照提供基础。

### 4.2 H1 随机模型

当前 H1 是复数频率响应，由 magnitude features 和 phase features 叠加得到。

Magnitude features 包括：

1. slope
2. ripple
3. notch/bump
4. edge rolloff
5. measurement noise

Phase features 包括：

1. linear phase delay
2. phase ripple
3. local phase distortion
4. group delay ripple
5. phase noise

这些 feature 的设计目的是使 H1 不是单一理想曲线，而是更接近真实硬件链路中可能出现的幅度和相位失真。

### 4.3 Bandwidth profile 设计

不同 bandwidth profile 会改变 H1 的频率范围、频率点数和随机失真强度。

| Profile | H1 frequency range | Num points |
|---|---:|---:|
| bw_500m | 5.75 GHz - 6.25 GHz | 1001 |
| bw_1g | 5.5 GHz - 6.5 GHz | 1001 |
| bw_2g | 5.0 GHz - 7.0 GHz | 1601 |
| bw_4g | 4.0 GHz - 8.0 GHz | 2001 |
| bw_8g | 2.0 GHz - 10.0 GHz | 3201 |

带宽越大，H1 随机模型中的 slope、ripple、notch/bump、edge rolloff、measurement noise 和 phase distortion 的允许范围整体增大。因此，profile 不是简单改变横轴范围，而是表示不同带宽和不同链路复杂度。

### 4.4 Seed case 设计

当前 sweep 中使用三组 seed case：

| Seed case | H1 seed | Behavior seed | QAM seed |
|---|---:|---:|---:|
| seed_a | 1040330957 | 1068822328 | 225239571 |
| seed_b | 297216821 | 953625061 | 453249928 |
| seed_c | 1571318759 | 1925817879 | 140927704 |

同一组 seed case 会在不同 bandwidth profile 下重复使用。这样可以同时观察：

1. 同一 bandwidth 下，不同随机 H1 的稳定性。
2. 同一 seed case 在不同 bandwidth 下的表现变化。

seed 数值本身不表示难度大小。真正决定 H1 难度的是 bandwidth profile 中设置的随机参数范围，以及该 seed 下具体抽样得到的 H1 形态。

### 4.5 Fixed-point 格式

当前 fixed-point sweep 使用三种格式：

| Format | Total bits | Fraction bits | Integer bits including sign |
|---|---:|---:|---:|
| Q2.14 | 16 | 14 | 2 |
| Q3.13 | 16 | 13 | 3 |
| Q4.12 | 16 | 12 | 4 |

小数位越多，量化精度越高，但整数范围越小；整数位越多，动态范围越大，但小数精度越低。因此 fixed-point 分析必须同时关注 quantization error 和 coefficient saturation。

---

## 5. 实验设置

本轮 sweep 设置如下：

| Dimension | Values |
|---|---|
| Bandwidth profile | bw_500m, bw_1g, bw_2g, bw_4g, bw_8g |
| Seed case | seed_a, seed_b, seed_c |
| FIR tap number | 64, 80, 96 |
| Regularization | 1e-4 |
| Fixed-point format | Q2.14, Q3.13, Q4.12 |

总组合数为：

```text
5 × 3 × 3 × 1 × 3 = 135
```

主要输出目录为：

```text
sweep_result/bandwidth_profile_seed_sweep/
```

主要结果文件包括：

```text
sweep_summary.csv
sweep_best_combos.csv
sweep_group_summary.csv
sweep_analysis_report.md
```

---

## 6. 实验结果

### 6.1 总体结果

| Metric | Result |
|---|---:|
| Total combos | 135 |
| Fixed dense ripple pass | 101 / 135 |
| Fixed multi-tone behavior pass | 122 / 135 |
| Saturated combos | 3 / 135 |

结果表明，在当前 H1 随机模型下，大多数组合可以完成有效补偿。但 dense ripple 通过数明显少于 behavior ripple，说明 dense grid 检查更严格，更适合作为主要 pass/fail 指标。

### 6.2 Bandwidth profile 结果

| Profile | Bandwidth | Dense pass | Behavior pass | Saturated | Best dense ripple | Best behavior ripple | Best QAM mag-only EVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| bw_500m | 500 MHz | 27/27 | 27/27 | 0 | 0.036109 dB | 0.027230 dB | 0.080535% |
| bw_1g | 1 GHz | 10/27 | 18/27 | 1 | 0.030020 dB | 0.017024 dB | 0.048134% |
| bw_2g | 2 GHz | 20/27 | 23/27 | 2 | 0.025346 dB | 0.014714 dB | 0.037186% |
| bw_4g | 4 GHz | 26/27 | 27/27 | 0 | 0.030839 dB | 0.021415 dB | 0.040919% |
| bw_8g | 8 GHz | 18/27 | 27/27 | 0 | 0.034844 dB | 0.023211 dB | 0.048770% |

`bw_500m` 最稳定，所有组合均通过 dense 和 behavior 检查。`bw_1g` 是当前最弱的 profile，主要受 `seed_c` 影响。`bw_8g` behavior 全部通过但 dense pass 为 18/27，说明 multi-tone 验证可能没有采样到最坏频点。

![Bandwidth vs fixed dense ripple](sweep_result/bandwidth_profile_seed_sweep/bandwidth_vs_fixed_dense_ripple.png)

### 6.3 Seed stability 结果

| Profile | Seed cases | Dense seed pass | Dense best / mean / worst | Behavior seed pass | Behavior best / mean / worst |
|---|---:|---:|---:|---:|---:|
| bw_500m | 3 | 3/3 | 0.036109 / 0.059759 / 0.088426 dB | 3/3 | 0.027230 / 0.046444 / 0.079353 dB |
| bw_1g | 3 | 2/3 | 0.030020 / 0.090470 / 0.141841 dB | 2/3 | 0.017024 / 0.065160 / 0.115332 dB |
| bw_2g | 3 | 3/3 | 0.025346 / 0.052314 / 0.074132 dB | 3/3 | 0.014714 / 0.029753 / 0.039097 dB |
| bw_4g | 3 | 3/3 | 0.030839 / 0.054029 / 0.082665 dB | 3/3 | 0.021415 / 0.039886 / 0.061975 dB |
| bw_8g | 3 | 2/3 | 0.034844 / 0.072167 / 0.119354 dB | 3/3 | 0.023211 / 0.044269 / 0.069286 dB |

这里的 `Dense seed pass` 表示某个 profile 下，每个 seed case 是否至少有一个配置能够通过 dense 0.1 dB 指标。结果显示，`bw_500m`、`bw_2g`、`bw_4g` 的 3 个 seed case 均可找到通过配置；`bw_1g` 和 `bw_8g` 只有 2/3 个 seed case 可找到通过配置。

### 6.4 Tap number 结果

| Tap num | Combos | Dense pass | Behavior pass | Saturated | Best dense | Best behavior | Best QAM mag-only EVM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 45 | 33 | 39 | 1 | 0.026404 dB | 0.014714 dB | 0.038317% |
| 80 | 45 | 31 | 41 | 1 | 0.025346 dB | 0.014810 dB | 0.037186% |
| 96 | 45 | 37 | 42 | 1 | 0.028876 dB | 0.016930 dB | 0.038309% |

96 tap 的 dense pass 数最高，整体最稳。但 64 tap 在多数场景下也能通过，因此可作为硬件资源 baseline；96 tap 可作为性能 upper bound。

![Dense ripple by tap](sweep_result/bandwidth_profile_seed_sweep/sweep_fixed_dense_ripple_by_tap.png)

### 6.5 Fixed-point 结果

| Fixed format | Combos | Dense pass | Behavior pass | Saturated | Best dense | Best behavior | Best QAM mag-only EVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2.14 | 45 | 34 | 40 | 3 | 0.025346 dB | 0.014714 dB | 0.037186% |
| Q3.13 | 45 | 35 | 41 | 0 | 0.028280 dB | 0.014810 dB | 0.038401% |
| Q4.12 | 45 | 32 | 41 | 0 | 0.029452 dB | 0.021169 dB | 0.054818% |

`Q2.14` 精度最高，但整数范围较小，出现 3 次 saturation。`Q3.13` 没有 saturation，同时 dense pass 数最高，因此是当前更稳妥的 fixed-point baseline。

出现 saturation 的组合为：

| Combo | Saturation count | Fixed dense ripple |
|---|---:|---:|
| bw_1g_seed_c_tap096_reg1em04_q2_14 | 6 | 2.899307 dB |
| bw_2g_seed_c_tap064_reg1em04_q2_14 | 4 | 74.681918 dB |
| bw_2g_seed_c_tap080_reg1em04_q2_14 | 2 | 67.807851 dB |

这些失败 case 说明 fixed-point 分析不能只看小数精度，还必须检查系数动态范围。

![Coefficient scale and saturation](sweep_result/bandwidth_profile_seed_sweep/sweep_saturation_and_coeff_range.png)

---

## 7. 结果讨论

### 7.1 Dense ripple 与 behavior ripple

Dense ripple 是在完整频率网格上计算 `|H1(f) * H2_fixed(f)|` 的残余幅度起伏。Behavior ripple 是使用有限个 multi-tone 时域输入测得的 tone amplitude ripple。

本轮实验中：

```text
dense pass = 101 / 135
behavior pass = 122 / 135
```

这说明 behavior ripple 通常比 dense ripple 更乐观。原因是 multi-tone 只采样有限频点，可能没有覆盖到 Htotal 的最坏点。因此本文将 dense ripple 作为主 pass/fail 指标，将 behavior ripple 作为行为级验证指标。

### 7.2 QAM magnitude-only EVM 的定位

QAM magnitude-only EVM 用于辅助观察幅度补偿对宽带调制信号的影响，但它不替代 dense ripple。当前 QAM 模块不是完整通信链路 EVM 评估，也不覆盖 L1-09 phase/group delay correction。因此，本文只将其作为 L1-08 magnitude equalization 的辅助指标。

![QAM EVM by tap](sweep_result/bandwidth_profile_seed_sweep/sweep_qam_evm_by_tap.png)

### 7.3 当前仿真的局限

当前 H1 由人工随机模型生成，虽然包含 slope、ripple、notch/bump、edge rolloff、noise 和 phase distortion，但仍不等价于真实 VNA 或硬件 NVM 数据。因此当前结论应理解为行为级算法验证结果，而不是最终硬件性能保证。

后续如果可以获得真实硬件 H1 数据，应将随机 H1 替换为实测 H1，并复用同一 pipeline 进行验证。

---

## 8. 结论

本文构建并验证了 L1-08 幅频 FIR 均衡算法的行为级仿真流程。实验结果表明：

1. L1-08 的基本补偿思路是可行的。
2. 在当前随机 H1 模型下，大多数 profile/seed 组合可以通过 0.1 dB dense ripple 指标。
3. 单个 seed 的结果不足以说明鲁棒性，必须保留 multiple-seed sweep。
4. `Q3.13` 是当前更稳妥的 fixed-point baseline，因为它没有 saturation 且 dense pass 数最高。
5. 96 tap 整体最稳，64 tap 可作为硬件资源 baseline。
6. Dense ripple 应作为主要判据，behavior ripple 和 QAM magnitude-only EVM 作为辅助验证。

因此，当前阶段已经达成导师提出的行为级仿真验证目标，但后续仍需进一步扩展 regularization sweep、固定配置跨 seed 鲁棒性分析，以及真实硬件 H1 数据验证。

---

## 9. 后续工作

建议下一阶段继续完成以下工作：

1. 增加 regularization sweep，例如 `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`。
2. 对 `tap064 + Q3.13` 与 `tap096 + Q3.13` 做重点对比。
3. 增加“同一 tap/fixed-point 配置跨所有 profile/seed 的鲁棒性表”。
4. 如果可以获得真实硬件数据，使用真实 H1 替代随机 H1。
5. 在 L1-08 magnitude equalization 稳定后，再进入 L1-09 phase/group delay correction 或 RTL 对照。

---

## 附录 A. 程序结构

| Program | Function |
|---|---|
| `L1-08_sim/H1_full_combined_random_generator.py` | 生成随机复数 H1 |
| `L1-08_sim/H2_target_generator.py` | 根据 H1 magnitude 生成 H2_target |
| `L1-08_sim/H2_fir_designer.py` | 设计 real linear-phase FIR |
| `L1-08_sim/H2_fixed_point_quantizer.py` | 进行 fixed-point 系数量化 |
| `L1-08_sim/L1_08_behavior_sim.py` | multi-tone complex I/Q 行为级验证 |
| `L1-08_sim/L1_08_qam_evm_sim.py` | QAM magnitude-only EVM 辅助验证 |
| `sweep_test/run_sweep.py` | 批量运行 sweep |
| `sweep_test/analyze_sweep_results.py` | 生成 sweep 分析报告和图表 |

---

## 附录 B. 复现实验命令

运行 sweep：

```powershell
& 'C:\Users\CodexSandboxOffline\.codex\.sandbox\cwd\668082bd16c2c5ea\L1-08_sim\.venv\Scripts\python.exe' sweep_test/run_sweep.py --config sweep_test/config.json
```

分析 sweep：

```powershell
& 'C:\Users\CodexSandboxOffline\.codex\.sandbox\cwd\668082bd16c2c5ea\L1-08_sim\.venv\Scripts\python.exe' sweep_test/analyze_sweep_results.py --config sweep_test/config.json
```

主要输出：

```text
sweep_result/bandwidth_profile_seed_sweep/sweep_summary.csv
sweep_result/bandwidth_profile_seed_sweep/sweep_best_combos.csv
sweep_result/bandwidth_profile_seed_sweep/sweep_group_summary.csv
sweep_result/bandwidth_profile_seed_sweep/sweep_analysis_report.md
```
