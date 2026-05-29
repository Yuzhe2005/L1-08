# L1-08 幅频 FIR 均衡算法行为级仿真报告

## 1. 当前结论

本阶段已经根据导师反馈，把原来只针对少量随机 H1 的单点验证，改成了更接近工程验证的行为级 sweep：

```text
5 个 bandwidth profile
× 3 组 seed case
× 3 个 FIR tap 数
× 1 个 regularization
× 3 个 fixed-point format
= 135 组组合
```

当前 135 组结果说明：

1. L1-08 的基本算法链路是正确、可运行的：随机 H1 生成、H2 target 生成、FIR 设计、fixed-point 量化、multi-tone 行为级验证、QAM 辅助验证都已经闭环。
2. 在当前随机 H1 模型下，fixed dense ripple 通过 `0.1 dB` 指标的数量为 `101 / 135`。
3. multi-tone behavior ripple 通过 `0.1 dB` 指标的数量为 `122 / 135`。
4. fixed-point saturation 出现 `3 / 135` 次，全部发生在 `Q2.14` 格式。
5. `Q3.13` 是当前更稳妥的 fixed-point 格式候选，因为它没有 saturation，同时 dense pass 数最高。
6. 96 tap 整体最稳，但 64 tap 在部分 bandwidth/seed 下也可以通过，因此后续需要根据硬件资源和性能目标做 tradeoff。

本报告的核心判断是：

> 当前仿真已经能支撑导师要求的“行为级算法验证”。下一步不应该只展示某一个好 seed，而应该继续用 profile + seed case 的方式说明算法在不同带宽和不同随机链路条件下的稳定性。

---

## 2. 导师反馈对应的修改方向

导师的主要建议可以概括为三点：

1. **先做行为级仿真**
   - 先验证算法目标和方向是否正确。
   - 不急着进入 RTL。
   - 通过仿真确认 H1 失真、H2 补偿、fixed-point 量化后的效果。

2. **模拟模块要清晰**
   - H1 随机生成、H2 target、FIR 设计、fixed-point、behavior simulation、QAM simulation 分开。
   - 每个模块输入输出明确。
   - 尽量使用对象化结构，方便后续扩展。

3. **仿真过程要能形成报告**
   - 不能只给代码和图片。
   - 需要记录仿真维度、seed、bandwidth、参数组合、结果和失败原因。
   - 要说明哪些结果可靠，哪些还只是当前模型下的初步观察。

当前程序已经按这个方向改进：

| 导师建议 | 当前实现 |
|---|---|
| 行为级仿真先跑通 | 已完成 H1 -> H2 target -> FIR -> fixed-point -> behavior/QAM 验证 |
| 模块清晰 | `L1-08_sim/` 下每个 stage 独立 |
| 随机输入可复现 | 使用 `seed_cases` 控制 H1、behavior、QAM 三类随机过程 |
| 不只看 1 GHz | 使用 `bw_500m, bw_1g, bw_2g, bw_4g, bw_8g` 五个 profile |
| 分析结果可汇报 | `sweep_analysis_report.md`、`sweep_summary.csv`、图表自动生成 |

---

## 3. 当前仿真结构

当前 pipeline 是：

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

对应程序：

| Stage | Program | Function |
|---|---|---|
| H1 random generation | `L1-08_sim/H1_full_combined_random_generator.py` | 生成幅度和相位随机 H1 |
| H2 target | `L1-08_sim/H2_target_generator.py` | 根据 H1 magnitude 生成反向补偿目标 |
| FIR design | `L1-08_sim/H2_fir_designer.py` | 设计 real linear-phase FIR |
| Fixed-point | `L1-08_sim/H2_fixed_point_quantizer.py` | 模拟有限 bit 系数量化 |
| Behavior simulation | `L1-08_sim/L1_08_behavior_sim.py` | 用 multi-tone complex I/Q 信号验证 |
| QAM simulation | `L1-08_sim/L1_08_qam_evm_sim.py` | 用 QAM-loaded I/Q 信号做辅助验证 |
| Sweep runner | `sweep_test/run_sweep.py` | 批量运行不同配置 |
| Sweep analysis | `sweep_test/analyze_sweep_results.py` | 汇总表格、图和 markdown 报告 |

---

## 4. H1 随机生成方式

当前 H1 是复数频率响应：

```text
H1(f) = H1_magnitude(f) * exp(j * H1_phase(f))
```

H1 random generator 由多个 single feature 叠加得到：

### 4.1 Magnitude features

1. slope
2. ripple
3. notch / bump
4. edge rolloff
5. measurement noise

这些 feature 叠加后得到 `magnitude_combined.csv`。

### 4.2 Phase features

1. linear phase delay
2. phase ripple
3. local phase distortion
4. group delay ripple
5. phase noise

这些 feature 叠加后得到 `phase_combined.csv`。

### 4.3 Together H1

最后把 magnitude 和 phase 合并：

```text
together.csv = magnitude_combined + phase_combined
```

需要注意：L1-08 当前主要补偿的是 magnitude ripple；phase 目前主要用于让 H1 更接近真实链路，并为后续 L1-09 phase / group delay 修复预留基础。

---

## 5. Bandwidth profile 如何影响 H1 generation

当前不是简单把同一个 H1 横向拉宽，而是让不同 bandwidth profile 对应不同的频率范围和 H1 随机失真强度。

| Profile | H1 frequency range | Num points |
|---|---:|---:|
| `bw_500m` | 5.75 GHz - 6.25 GHz | 1001 |
| `bw_1g` | 5.5 GHz - 6.5 GHz | 1001 |
| `bw_2g` | 5.0 GHz - 7.0 GHz | 1601 |
| `bw_4g` | 4.0 GHz - 8.0 GHz | 2001 |
| `bw_8g` | 2.0 GHz - 10.0 GHz | 3201 |

带宽越大，H1 random model 的失真范围整体越强：

1. slope 最大值更大。
2. ripple 幅度更大，cycles 更多。
3. notch/bump 更深，数量可能更多。
4. edge rolloff 更强。
5. measurement noise 更大。
6. phase delay、phase ripple、local phase distortion、group delay ripple、phase noise 也同步增强。

因此 profile 的含义是：

> bandwidth profile 控制链路难度；seed case 控制同一难度下的随机样本。

---

## 6. Seed case 的含义

当前 `sweep_test/config.json` 中配置了 3 组 seed case：

| Seed case | H1 seed | Behavior seed | QAM seed |
|---|---:|---:|---:|
| `seed_a` | 1040330957 | 1068822328 | 225239571 |
| `seed_b` | 297216821 | 953625061 | 453249928 |
| `seed_c` | 1571318759 | 1925817879 | 140927704 |

同一个 seed case 会在所有 bandwidth profile 下重复使用。

这不代表不同 bandwidth 下生成完全相同的 H1，因为 profile 会改变频率范围和随机参数 limit。它的意义是：

```text
同一个 seed_case = 同一组随机样本编号
不同 profile = 不同带宽和不同难度条件
```

这样可以分析两个方向：

1. **同一个 bandwidth，不同 seed 是否稳定。**
2. **同一个 seed case，不同 bandwidth 下是否越来越难。**

---

## 7. Sweep 设置

本轮 sweep 使用：

| Dimension | Values |
|---|---|
| bandwidth profile | `bw_500m`, `bw_1g`, `bw_2g`, `bw_4g`, `bw_8g` |
| seed case | `seed_a`, `seed_b`, `seed_c` |
| tap_num | `64`, `80`, `96` |
| regularization | `1e-4` |
| fixed-point format | `Q2.14`, `Q3.13`, `Q4.12` |

总组合数：

```text
5 * 3 * 3 * 1 * 3 = 135
```

结果目录：

```text
sweep_result/bandwidth_profile_seed_sweep/
```

主要分析文件：

```text
sweep_summary.csv
sweep_best_combos.csv
sweep_group_summary.csv
sweep_analysis_report.md
```

---

## 8. 总体结果

| Metric | Result |
|---|---:|
| Total combos | 135 |
| Fixed dense ripple pass | 101 / 135 |
| Fixed multi-tone behavior pass | 122 / 135 |
| Saturated combos | 3 / 135 |

这里最重要的是 fixed dense ripple。

原因是 dense ripple 是在完整 H1 频率网格上检查 `H1 * H2_fixed` 的残余幅度起伏，覆盖点比 multi-tone behavior 更密。因此它比 behavior ripple 更严格。

---

## 9. 按 bandwidth profile 分析

| Profile | Bandwidth | Dense pass | Behavior pass | Saturated | Best dense ripple | Best behavior ripple | Best QAM mag-only EVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bw_500m` | 500 MHz | 27/27 | 27/27 | 0 | 0.036109 dB | 0.027230 dB | 0.080535% |
| `bw_1g` | 1 GHz | 10/27 | 18/27 | 1 | 0.030020 dB | 0.017024 dB | 0.048134% |
| `bw_2g` | 2 GHz | 20/27 | 23/27 | 2 | 0.025346 dB | 0.014714 dB | 0.037186% |
| `bw_4g` | 4 GHz | 26/27 | 27/27 | 0 | 0.030839 dB | 0.021415 dB | 0.040919% |
| `bw_8g` | 8 GHz | 18/27 | 27/27 | 0 | 0.034844 dB | 0.023211 dB | 0.048770% |

结论：

1. `bw_500m` 最稳定，所有组合 dense 和 behavior 都通过。
2. `bw_1g` 是本轮里最弱的 profile，主要问题来自 `seed_c`。
3. `bw_2g` 虽然整体可补偿，但 `seed_c + Q2.14` 出现了 fixed-point saturation。
4. `bw_4g` 表现很好，说明更宽 bandwidth 并不必然更差，关键还取决于具体随机 H1 形状。
5. `bw_8g` 的 behavior 全部通过，但 dense 只有 18/27，说明 multi-tone 可能没有采到最坏频点。

![Bandwidth vs fixed dense ripple](sweep_result/bandwidth_profile_seed_sweep/bandwidth_vs_fixed_dense_ripple.png)

---

## 10. Seed stability 分析

| Profile | Seed cases | Dense seed pass | Dense best / mean / worst | Behavior seed pass | Behavior best / mean / worst |
|---|---:|---:|---:|---:|---:|
| `bw_500m` | 3 | 3/3 | 0.036109 / 0.059759 / 0.088426 dB | 3/3 | 0.027230 / 0.046444 / 0.079353 dB |
| `bw_1g` | 3 | 2/3 | 0.030020 / 0.090470 / 0.141841 dB | 2/3 | 0.017024 / 0.065160 / 0.115332 dB |
| `bw_2g` | 3 | 3/3 | 0.025346 / 0.052314 / 0.074132 dB | 3/3 | 0.014714 / 0.029753 / 0.039097 dB |
| `bw_4g` | 3 | 3/3 | 0.030839 / 0.054029 / 0.082665 dB | 3/3 | 0.021415 / 0.039886 / 0.061975 dB |
| `bw_8g` | 3 | 2/3 | 0.034844 / 0.072167 / 0.119354 dB | 3/3 | 0.023211 / 0.044269 / 0.069286 dB |

这里的 `Dense seed pass` 不是统计所有 combo，而是看每个 seed case 在该 profile 下是否至少有一个组合能通过 dense `0.1 dB` 指标。

这个表回答的是：

> 同一个 bandwidth 下，换不同随机 H1 后，算法是否仍然能找到有效补偿配置。

结论：

1. `bw_500m`, `bw_2g`, `bw_4g` 的 3 个 seed case 都能找到 dense pass 配置。
2. `bw_1g` 只有 2/3 个 seed case 能找到 dense pass 配置。
3. `bw_8g` 只有 2/3 个 seed case 能找到 dense pass 配置。
4. 所以当前算法不是只对一个 seed 有效，但也不是对所有随机样本都完全稳定。

---

## 11. 按 seed case 分析

| Seed case | Combos | Dense pass | Behavior pass | Saturated | Best dense | Best behavior | Best QAM mag-only EVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seed_a` | 45 | 45 | 45 | 0 | 0.025346 dB | 0.014714 dB | 0.037186% |
| `seed_b` | 45 | 26 | 45 | 0 | 0.054740 dB | 0.032750 dB | 0.088620% |
| `seed_c` | 45 | 30 | 32 | 3 | 0.048583 dB | 0.035446 dB | 0.071743% |

结论：

1. `seed_a` 是本轮最容易的随机样本，所有 profile 和配置下都通过 dense/behavior。
2. `seed_b` 的 behavior 都通过，但 dense 只有 26/45，说明 dense 检查更严格。
3. `seed_c` 出现 3 次 saturation，是 fixed-point 风险的主要来源。

---

## 12. 按 tap_num 分析

| Tap num | Combos | Dense pass | Behavior pass | Saturated | Best dense | Best behavior | Best QAM mag-only EVM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 45 | 33 | 39 | 1 | 0.026404 dB | 0.014714 dB | 0.038317% |
| 80 | 45 | 31 | 41 | 1 | 0.025346 dB | 0.014810 dB | 0.037186% |
| 96 | 45 | 37 | 42 | 1 | 0.028876 dB | 0.016930 dB | 0.038309% |

从 pass 数看，96 tap 最稳：

```text
96 tap dense pass: 37 / 45
64 tap dense pass: 33 / 45
80 tap dense pass: 31 / 45
```

但是 64 tap 并不是完全不可用，它在很多 profile/seed 下也能通过。由于算法交付物更倾向 64-tap FIR，后续可以把 64 tap 作为硬件资源 baseline，把 96 tap 作为性能 upper bound。

![Dense ripple by tap](sweep_result/bandwidth_profile_seed_sweep/sweep_fixed_dense_ripple_by_tap.png)

---

## 13. Fixed-point 分析

| Fixed format | Combos | Dense pass | Behavior pass | Saturated | Best dense | Best behavior | Best QAM mag-only EVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Q2.14` | 45 | 34 | 40 | 3 | 0.025346 dB | 0.014714 dB | 0.037186% |
| `Q3.13` | 45 | 35 | 41 | 0 | 0.028280 dB | 0.014810 dB | 0.038401% |
| `Q4.12` | 45 | 32 | 41 | 0 | 0.029452 dB | 0.021169 dB | 0.054818% |

结论：

1. `Q2.14` 小数精度最高，但整数范围较小，因此出现了 3 次 saturation。
2. `Q3.13` 没有 saturation，同时 dense pass 数最高，是当前最合理的 fixed-point baseline。
3. `Q4.12` 范围更大，但小数精度更低，dense pass 数略低。

当前出现 saturation 的组合为：

| Combo | Saturation count | Fixed dense ripple |
|---|---:|---:|
| `bw_1g_seed_c_tap096_reg1em04_q2_14` | 6 | 2.899307 dB |
| `bw_2g_seed_c_tap064_reg1em04_q2_14` | 4 | 74.681918 dB |
| `bw_2g_seed_c_tap080_reg1em04_q2_14` | 2 | 67.807851 dB |

这些组合说明 fixed-point 不能只看 quantization 精度，也必须检查 coefficient dynamic range。

![Coefficient scale and saturation](sweep_result/bandwidth_profile_seed_sweep/sweep_saturation_and_coeff_range.png)

---

## 14. Dense ripple 和 behavior ripple 的区别

### 14.1 Dense ripple

Dense ripple 直接在 H1 frequency grid 上计算：

```text
|H1(f) * H2_fixed(f)|
```

它回答的是：

> 完整频带内，补偿后最坏幅度起伏是多少？

这是 L1-08 当前最严格、最核心的 pass/fail 指标。

### 14.2 Behavior ripple

Behavior ripple 是用 multi-tone complex I/Q 时域信号仿真后测得的 tone amplitude ripple。

它回答的是：

> 对当前这一组 multi-tone 输入，输出 tone 的幅度是否平坦？

### 14.3 为什么 behavior 可能比 dense 好

multi-tone 只检查有限个 tone。如果 Htotal 的最坏 ripple 出现在 tone 之间，behavior ripple 可能看起来比 dense ripple 更好。

本轮结果中：

```text
dense pass:   101 / 135
behavior pass: 122 / 135
```

这说明 behavior 验证是有价值的，但主判断仍然应该优先看 dense ripple。

![Behavior ripple by tap](sweep_result/bandwidth_profile_seed_sweep/sweep_behavior_ripple_by_tap.png)

---

## 15. QAM magnitude-only EVM 的作用

当前 QAM simulation 是辅助验证，不替代 dense ripple 和 multi-tone behavior。

它的意义是：

1. 用更接近宽带调制信号的输入观察幅度补偿效果。
2. 辅助说明 L1-08 对调制信号幅度误差的改善趋势。
3. 为后续更完整的 EVM / L1-09 phase correction 留接口。

当前报告使用的是 magnitude-only EVM，而不是完整通信链路意义上的 full EVM。原因是 L1-08 的目标主要是 magnitude equalization，不是 phase/group-delay correction。

![QAM EVM by tap](sweep_result/bandwidth_profile_seed_sweep/sweep_qam_evm_by_tap.png)

---

## 16. 当前最重要的工程判断

### 16.1 当前算法流程合理

从 135 组 sweep 看，算法能在多数 profile/seed 下把 residual magnitude ripple 压到 `0.1 dB` 内。

这说明：

```text
H2_target = inverse magnitude of H1
real linear-phase FIR approximation
fixed-point coefficient quantization
```

这个方向是可行的。

### 16.2 不能只用单个 seed 下结论

如果只看 `seed_a`，会得到过于乐观的结论，因为 `seed_a` 的 45 组全都通过。

但是 `seed_b` 和 `seed_c` 暴露出：

1. 有些随机 H1 对 dense grid 更难。
2. 有些 fixed-point format 会 saturation。
3. behavior pass 不代表 dense 一定 pass。

因此后续报告和程序都应该保留 `seed_cases` 维度。

### 16.3 Q3.13 是当前更稳妥的 fixed-point 起点

`Q2.14` 精度高，但是已经出现 saturation。

`Q3.13` 没有 saturation，同时 dense pass 数最高。因此下一轮如果要选一个默认 fixed-point format，可以优先用：

```text
coeff_total_bits = 16
coeff_frac_bits = 13
format = Q3.13
```

### 16.4 96 tap 是当前性能更稳的候选

96 tap 在当前 sweep 中 dense pass 数最高。

但是如果硬件目标必须接近 64 tap，则需要继续优化：

1. 调整 FIR 设计方法。
2. 调整 regularization sweep。
3. 对 H1 random model 的真实程度做校准。
4. 用真实 VNA/NVM 数据代替人工随机 H1。

---

## 17. 当前报告结论

本阶段已经完成了导师要求的行为级仿真框架，并且把验证从单个随机输入扩展到了 bandwidth profile 和 seed case 两个维度。

当前结论可以概括为：

```text
L1-08 FIR magnitude equalization 的算法流程是合理的；
在当前随机 H1 模型下，大多数 profile/seed 可以通过 0.1 dB dense ripple 指标；
但 fixed-point format 和随机 H1 复杂度会显著影响结果；
因此后续需要继续做跨 seed、跨 bandwidth 的鲁棒性验证，而不是只汇报单个最好 case。
```

推荐下一阶段工作：

1. 增加 regularization sweep，例如 `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`。
2. 对 `tap096 + Q3.13` 和 `tap064 + Q3.13` 做单独对比。
3. 增加“同一个 tap/fixed format 在所有 profile/seed 下的鲁棒性表”。
4. 如果导师或项目能提供真实硬件数据，优先用真实 H1 替代 random H1。
5. 后续再考虑 RTL 对照和 L1-09 phase/group-delay 修复。

---

## 18. 如何复现实验

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
