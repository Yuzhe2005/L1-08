# L1-08 Pipeline 使用说明

本文档说明 `Rigol` 项目中 L1-08 行为级仿真的代码结构、配置入口、运行流程、每一步输入输出、sweep test 用法，以及每类结果文件应该如何理解。

当前项目根目录：

```text
D:\桌面\Rigol
```

L1-08 的核心目标是：模拟硬件前端 `H1` 在 3.5~4.5 GHz IF 频段内造成的幅频 ripple，然后设计一个 real linear-phase FIR 作为 `H2`，让总响应：

```text
Htotal(f) = H1(f) * H2(f)
```

在目标带宽内尽量平坦。

---

## 1. 项目目录结构

当前项目主要文件夹如下：

```text
Rigol/
├── 03-频谱仪算法栈交付物.html
├── L1-08_algorithm_design_report.md
├── config_input.json         # H1 / behavior / QAM 输入参数
├── config_base_plan.json     # H2 FIR / fixed-point / L1-09 / run 开关
├── config_plan_b.json        # Plan B 独立设计参数
├── README_pipeline.md
├── L1-08_planning/
│   ├── L1-08.md
│   └── L1-08.docx
├── config_base_plan_sweep.json   # Base Plan sweep 参数
├── config_plan_b_sweep.json      # Plan B sweep 参数
├── shared_sim/
│   ├── h1_source.py          # shared H1 generation (CLI entry)
│   ├── config.py io_utils.py run_summary.py signal_utils.py
│   ├── h1_common.py qam_utils.py behavior_utils.py
│   └── magnitude/ phase/
├── L1-08+L1-09_sim_base_plan/
│   ├── run_full_l1_08_l1_09_pipeline.py
│   ├── L1-08_sim/
│   │   ├── run_all_pipeline.py
│   │   └── ...
│   ├── L1_09_sim/
│   └── sweep_test/
│       ├── run_sweep.py
│       └── analyze_sweep_results.py
├── L1-08+L1-09_sim_planB/
│   ├── run_all_plan_b_pipeline.py
│   └── sweep_test/
│       ├── run_plan_b_sweep.py
│       └── analyze_plan_b_sweep_results.py
├── experimental/             # archived planC / past_version
├── data/ graph/
└── sweep_result/             # 两套 sweep 的统一输出目录
```

其中最重要的是：

| 路径 | 作用 |
|---|---|
| `config_input.json` | H1、behavior、QAM 等共享输入配置（含 `profiles`） |
| `config_base_plan.json` | Base Plan：H2 FIR、定点、L1-09、`run` 阶段开关 |
| `config_plan_b.json` | Plan B：complex FIR 设计、定点、阶段开关、`input.run_dir` |
| `L1-08+L1-09_sim_base_plan/L1-08_sim/run_all_pipeline.py` | 一键运行完整 L1-08 pipeline（零 CLI 参数） |
| `L1-08+L1-09_sim_base_plan/run_full_l1_08_l1_09_pipeline.py` | L1-08 + L1-09 全链路 |
| `data/` | 单次 pipeline 的 CSV/JSON 输出 |
| `graph/` | 单次 pipeline 的 PNG 图像输出 |
| `config_base_plan_sweep.json` | Base Plan sweep 参数 |
| `config_plan_b_sweep.json` | Plan B sweep 参数 |
| `L1-08+L1-09_sim_base_plan/sweep_test/run_sweep.py` | Base Plan 批量扫描 |
| `L1-08+L1-09_sim_planB/sweep_test/run_plan_b_sweep.py` | Plan B 批量扫描 |
| `sweep_result/` | 两套 sweep 的统一输出目录 |
| `L1-08_algorithm_design_report.md` | 当前算法模拟报告 |

---

## 2. 当前仿真的信号设定

当前仿真按导师回复后的理解执行：L1-08 补偿发生在 DDC 之前，所以处理对象是 **complex I/Q IF**，不是 centered complex baseband。

主频率设置来自 `config_input.json` 的 `active.common`：

```text
采样率 Fs = 12 GHz
H1 频率建模范围 = 3.5 GHz ~ 4.5 GHz
H1 频率点数量 = 1001
multi-tone 输入范围 = 3.55 GHz ~ 4.45 GHz
QAM-loaded 输入范围 = 3.55 GHz ~ 4.45 GHz
```

为什么输入范围是 `3.55~4.45 GHz`，而不是刚好 `3.5~4.5 GHz`：

```text
H1 建模范围留作完整硬件响应边界
输入 tone / QAM bins 稍微向内收一点，避免刚好落在边界点
```

当前 L1-08 只主动补偿 magnitude ripple。相位和 group delay 图会被画出来，但主要用于说明 real linear-phase FIR 本身只引入 constant group delay，不负责修 H1 的非线性相位；这部分更接近 L1-09 的范围。

---

## 3. 主配置文件

项目根目录使用三个 JSON，职责分离：

| 文件 | 内容 |
|---|---|
| `config_input.json` | `active` + `profiles`：common / h1 / behavior / qam_evm；`selected_profile` 选带宽 profile |
| `config_base_plan.json` | `active`：h2_fir / fixed_point / l1_09 / run（skip 开关、validation 模式） |
| `config_plan_b.json` | `active`：input.run_dir / design / fixed_point / stages |

用户面向的 pipeline 入口均**无需 CLI 参数**；改参数请编辑对应 JSON。

### 3.1 active

`active` 是单次 pipeline 实际使用的配置。

主要字段：

| 字段 | 当前作用 |
|---|---|
| `common.fs_hz` | 采样率，当前 12 GHz |
| `h1.seed` | H1 随机生成 seed |
| `h1_random_model` | H1 magnitude / phase 随机模型参数 |
| `h2_fir.tap_num` | 常规 pipeline FIR tap 数 |
| `h2_fir.regularization` | LS/ridge regularization 强度 |
| `behavior.samples` | multi-tone 时域采样点数 |
| `behavior.settle_samples` | FIR 初始瞬态丢弃点数 |
| `behavior.tone_count` | multi-tone tone 数量 |
| `behavior.tone_min_hz` | multi-tone 起始频率 |
| `behavior.tone_max_hz` | multi-tone 结束频率 |
| `behavior.peak_amplitude` | 输入时域峰值归一化目标 |
| `behavior.seed` | multi-tone 随机相位 seed |
| `qam_evm.samples` | QAM/IFFT block 点数 |
| `qam_evm.freq_min_hz` | QAM bins 起始频率 |
| `qam_evm.freq_max_hz` | QAM bins 结束频率 |
| `qam_evm.qam_order` | QAM 阶数，当前 64-QAM |
| `qam_evm.peak_amplitude` | QAM 时域信号峰值归一化目标 |
| `qam_evm.seed` | QAM 随机 symbol seed |
| `fixed_point.coeff_total_bits` | FIR 系数定点总 bit 数 |
| `fixed_point.coeff_frac_bits` | FIR 系数定点小数 bit 数 |

### 3.2 sweep

`L1_08_experiment_config.json` 里的 `sweep` 是候选参数池，主要用于记录可能值得扫描的参数范围。

实际当前 sweep test 运行时使用的是：

```text
sweep_test_config.json
```

也就是说：

```text
单次 pipeline 看 L1_08_experiment_config.json 的 active
sweep test 看 sweep_test_config.json
```

---

## 4. 一键运行完整 pipeline

在 PowerShell 中进入项目根目录：

```powershell
cd D:\桌面\Rigol
```

运行完整 pipeline：

```powershell
python L1-08_sim\run_all_pipeline.py
```

完整运行顺序是：

```text
1. H1 random generation
2. H2 target generation
3. H2 FIR design
4. fixed-point coefficient quantization
5. multi-tone behavior simulation
6. QAM/EVM verification
```

如果只想跑 L1-08 主链路，跳过 QAM/EVM：

```powershell
python L1-08_sim\run_all_pipeline.py --skip-qam-evm
```

如果调试时只想跑到某一步：

```powershell
python L1-08_sim\run_all_pipeline.py --stop-after h2_fir_design
```

可选的 `--stop-after` stage 名称包括：

```text
h1_generation
h2_target_generation
h2_fir_design
fixed_point_coefficient_quantization
behavior_simulation
qam_evm_simulation
```

---

## 5. 单次 pipeline 输出位置

每次从 H1 generator 开始运行时，程序会新建一个 run folder。

典型 run name：

```text
h1_full_combined_random_YYYYMMDD_HHMMSS
```

对应输出位置：

```text
L1-08_sim/data/<run_name>/
graph/<run_name>/
```

其中：

```text
data/<run_name>/    保存 CSV 和 JSON
graph/<run_name>/ 保存 PNG 图像
```

当前如果 `L1-08_sim/data/` 和 `graph/` 是空的，这是正常的，因为之前做过清理。重新运行 `run_all_pipeline.py` 后会重新生成。

---

## 6. Pipeline 每一步详解

### 6.1 Step 1：H1 随机生成

脚本：

```text
shared_sim/h1_source.py
```

作用：

生成模拟硬件前端通道 `H1`，包括：

```text
H1 magnitude ripple
H1 phase distortion
```

当前频率轴：

```text
3.5 GHz ~ 4.5 GHz
1001 points
```

输入：

```text
L1_08_experiment_config.json
  active.h1.seed
  active.h1_random_model
```

主要输出：

```text
L1-08_sim/data/<run_name>/magnitude_combined.csv
L1-08_sim/data/<run_name>/phase_combined.csv
L1-08_sim/data/<run_name>/together.csv
L1-08_sim/data/<run_name>/run_summary.json
graph/<run_name>/magnitude_combined_magnitude.png
graph/<run_name>/phase_combined_phase.png
```

文件含义：

| 文件 | 含义 |
|---|---|
| `magnitude_combined.csv` | H1 的幅频响应，单位通常是 dB |
| `phase_combined.csv` | H1 的相频响应，单位通常是 rad |
| `together.csv` | H1 magnitude 和 phase 合并后的表 |
| `magnitude_combined_magnitude.png` | H1 magnitude 图 |
| `phase_combined_phase.png` | H1 phase 图 |

报告中用途：

说明硬件前端为什么需要补偿，也就是原始 `H1` 并不平坦。

---

### 6.2 Step 2：H2_target 生成

脚本：

```text
L1-08_sim/H2_target_generator.py
```

作用：

根据 `H1 magnitude` 生成理想反向补偿目标 `H2_target`。

核心思想：

```text
如果 H1 某个频点偏高，H2_target 就在该频点压低
如果 H1 某个频点偏低，H2_target 就在该频点抬高
```

理想情况下：

```text
|H1(f)| * |H2_target(f)| ≈ 常数
```

默认输入：

```text
最新的 L1-08_sim/data/<run_name>/magnitude_combined.csv
```

主要输出：

```text
L1-08_sim/data/<run_name>/h2_target.csv
graph/<run_name>/h2_target.png
```

报告中用途：

说明 L1-08 不是凭空设计 FIR，而是先根据硬件频响计算理想补偿目标。

---

### 6.3 Step 3：H2 FIR 设计

脚本：

```text
L1-08_sim/H2_fir_designer.py
```

作用：

用 real linear-phase FIR 拟合 `H2_target`，得到实际可实现的 `H2_actual`。

当前常规配置来自：

```text
L1_08_experiment_config.json
  active.h2_fir.tap_num
  active.h2_fir.regularization
  active.common.fs_hz
```

默认输入：

```text
L1-08_sim/data/<run_name>/h2_target.csv
```

主要输出：

```text
L1-08_sim/data/<run_name>/h2_fir_coefficients.csv
L1-08_sim/data/<run_name>/h2_actual_response.csv
graph/<run_name>/h2_fir_design.png
```

文件含义：

| 文件 | 含义 |
|---|---|
| `h2_fir_coefficients.csv` | float FIR 系数 |
| `h2_actual_response.csv` | float FIR 实际频响 |
| `h2_fir_design.png` | H1、H2_target、H2_actual、Htotal_actual 对比图 |

数学方法：

```text
real linear-phase FIR + least squares + ridge regularization
```

重点指标：

```text
ripple_before_db
ripple_after_db
meets_0p1db_target
max_abs_coeff
coeff_symmetry_max_error
```

报告中用途：

这是 L1-08 的核心算法设计步骤，用来说明 FIR 是否能把补偿后 residual ripple 压到 `0.1 dB` 以内。

---

### 6.4 Step 4：fixed-point 系数量化

脚本：

```text
L1-08_sim/H2_fixed_point_quantizer.py
```

作用：

把 float FIR 系数量化成 fixed-point FIR 系数，并重新计算 fixed-point FIR 的实际频响。

配置来自：

```text
L1_08_experiment_config.json
  active.fixed_point.coeff_total_bits
  active.fixed_point.coeff_frac_bits
```

默认输入：

```text
L1-08_sim/data/<run_name>/h2_fir_coefficients.csv
L1-08_sim/data/<run_name>/h2_target.csv
```

主要输出：

```text
L1-08_sim/data/<run_name>/h2_fir_coefficients_fixed.csv
L1-08_sim/data/<run_name>/h2_fixed_point_response.csv
graph/<run_name>/h2_fixed_point_quantization.png
```

文件含义：

| 文件 | 含义 |
|---|---|
| `h2_fir_coefficients_fixed.csv` | 定点量化后的 FIR 系数 |
| `h2_fixed_point_response.csv` | fixed-point FIR 频响 |
| `h2_fixed_point_quantization.png` | float/fixed 系数和频响对比 |

重点指标：

```text
saturation_count
max_abs_coeff_float
max_abs_coeff_fixed
coeff_max_abs_error
coeff_rms_error
ripple_after_fixed_db
meets_0p1db_target_fixed
```

报告中用途：

说明算法不仅 float 下能跑，还考虑了后续硬件定点实现的系数量化误差。

---

### 6.5 Step 5：multi-tone 行为级仿真

脚本：

```text
L1-08_sim/L1_08_behavior_sim.py
```

作用：

生成 complex I/Q multi-tone IF 输入，并在时域模拟：

```text
input
-> H1
-> float FIR
-> fixed-point FIR
```

当前默认配置：

```text
samples = 65536
settle_samples = 256
tone_count = 51
tone range = 3.55 GHz ~ 4.45 GHz
peak_amplitude = 0.8
```

默认输入：

```text
L1-08_sim/data/<run_name>/magnitude_combined.csv
L1-08_sim/data/<run_name>/phase_combined.csv
L1-08_sim/data/<run_name>/h2_fir_coefficients.csv
L1-08_sim/data/<run_name>/h2_fir_coefficients_fixed.csv
```

主要输出：

```text
L1-08_sim/data/<run_name>/input_iq.csv
L1-08_sim/data/<run_name>/after_h1_iq.csv
L1-08_sim/data/<run_name>/after_fir_iq.csv
L1-08_sim/data/<run_name>/after_fir_fixed_iq.csv
L1-08_sim/data/<run_name>/multitone_frequencies.csv
L1-08_sim/data/<run_name>/tone_amplitude_before_after.csv
graph/<run_name>/l1_08_behavior_multitone.png
graph/<run_name>/l1_08_behavior_phase_combined.png
```

文件含义：

| 文件 | 含义 |
|---|---|
| `input_iq.csv` | 输入 complex I/Q 时域信号 |
| `after_h1_iq.csv` | 输入经过 H1 后的 I/Q |
| `after_fir_iq.csv` | H1 后再经过 float FIR 的 I/Q |
| `after_fir_fixed_iq.csv` | H1 后再经过 fixed-point FIR 的 I/Q |
| `multitone_frequencies.csv` | 51 个 tone 的频率 |
| `tone_amplitude_before_after.csv` | 每个 tone 的幅度/相位测量结果 |
| `l1_08_behavior_multitone.png` | multi-tone 幅度补偿效果 |
| `l1_08_behavior_phase_combined.png` | phase 和 group delay 总结图 |

重点指标：

```text
ripple_after_h1_db
ripple_after_fir_db
ripple_after_fir_fixed_db
meets_0p1db_target
meets_0p1db_target_fixed
```

报告中用途：

这是最贴近算法交付物要求的核心行为级验证，因为交付物中 L1-08 的验证方法就是 multi-tone。

---

### 6.6 Step 6：QAM/EVM 辅助验证

脚本：

```text
L1-08_sim/L1_08_qam_evm_sim.py
```

作用：

生成 QAM-loaded complex I/Q IF 输入，并辅助观察幅频 ripple 对 QAM/EVM 的影响。

当前默认配置：

```text
samples = 65536
freq range = 3.55 GHz ~ 4.45 GHz
qam_order = 64
peak_amplitude = 0.8
max_constellation_points = 3000
```

生成方式：

```text
1. 找到 3.55~4.45 GHz 内的 FFT bins
2. 在这些 bins 上放随机 64-QAM symbol
3. IFFT 得到时域 complex I/Q IF 信号
4. 经过 H1、float FIR、fixed-point FIR
5. 计算 full EVM 和 magnitude-only EVM
```

主要输出：

```text
L1-08_sim/data/<run_name>/qam_input_iq.csv
L1-08_sim/data/<run_name>/qam_after_h1_iq.csv
L1-08_sim/data/<run_name>/qam_after_fir_iq.csv
L1-08_sim/data/<run_name>/qam_after_fir_fixed_iq.csv
L1-08_sim/data/<run_name>/qam_evm_summary.csv
L1-08_sim/data/<run_name>/qam_constellation_points.csv
graph/<run_name>/l1_08_qam_evm.png
```

重点指标：

```text
after_h1_evm_percent
after_float_fir_evm_percent
after_fixed_fir_evm_percent
after_h1_magnitude_only_evm_percent
after_float_fir_magnitude_only_evm_percent
after_fixed_fir_magnitude_only_evm_percent
```

注意：

`QAM/EVM` 是辅助验证，不是当前 L1-08 的主判据。L1-08 主要修 magnitude，full EVM 里包含 phase/group delay 影响，因此 full EVM 不应该直接等同于算法交付物里的 `EVM_LIN`。更接近 L1-08 目标的是：

```text
magnitude-only EVM
residual magnitude ripple
```

---

## 7. run_summary.json

每次 pipeline 会维护一个 summary 文件：

```text
L1-08_sim/data/<run_name>/run_summary.json
```

它会记录每个 stage 的关键信息和指标。

主要结构：

```text
run_name
data_dir
graph_dir
stages
  h1_generation
  h2_target_generation
  h2_fir_design
  fixed_point_coefficient_quantization
  behavior_simulation
  qam_evm_simulation
```

报告整理时，建议优先从 `run_summary.json` 读取关键数值，而不是手动从日志里抄。

---

## 8. 常规 pipeline 的结果怎么看

最重要的图：

| 图 | 看什么 |
|---|---|
| `magnitude_combined_magnitude.png` | H1 原始幅频 ripple |
| `phase_combined_phase.png` | H1 原始相频响应 |
| `h2_target.png` | 理想 H2 target 是否正确反向补偿 H1 |
| `h2_fir_design.png` | float FIR 后 Htotal 是否变平 |
| `h2_fixed_point_quantization.png` | fixed-point 后频响是否仍接近 float |
| `l1_08_behavior_multitone.png` | multi-tone 时域仿真中补偿前后幅度变化 |
| `l1_08_behavior_phase_combined.png` | FIR 是否表现为 constant group delay |
| `l1_08_qam_evm.png` | QAM/EVM 辅助结果 |

最重要的 CSV：

| CSV | 看什么 |
|---|---|
| `magnitude_combined.csv` | H1 magnitude 数据 |
| `phase_combined.csv` | H1 phase 数据 |
| `h2_target.csv` | 理想补偿目标 |
| `h2_fir_coefficients.csv` | float FIR 系数 |
| `h2_fir_coefficients_fixed.csv` | fixed-point FIR 系数 |
| `h2_actual_response.csv` | float FIR 频响 |
| `h2_fixed_point_response.csv` | fixed-point FIR 频响 |
| `tone_amplitude_before_after.csv` | multi-tone 每个 tone 的补偿前后幅度/相位 |
| `qam_evm_summary.csv` | QAM/EVM 数值摘要 |
| `run_summary.json` | 整个 run 的总摘要 |

---

## 9. Sweep test

如果要比较多个 tap 数、regularization 和 fixed-point 格式，需要运行 sweep test。

sweep 配置文件：

```text
sweep_test_config.json
```

当前 sweep 参数：

```text
tap_num: 64, 80, 96
regularization: 1e-4, 3e-4, 1e-3
coeff_total_bits: 16
coeff_frac_bits: 14, 13, 12
```

组合数量：

```text
3 tap choices * 3 regularization choices * 3 fixed-point choices = 27 combos
```

运行 sweep：

```powershell
python sweep_test\run_sweep.py
```

如果只想看会跑哪些 combo，不真正运行：

```powershell
python sweep_test\run_sweep.py --dry-run
```

当前 sweep 输出按 seed 分组：

```text
sweep_result/h1_<h1_seed>_behavior_<behavior_seed>_qam_<qam_seed>/
```

每个 combo 输出结构：

```text
sweep_result/<seed_folder>/<combo_folder>/
├── data/
├── graph/
├── logs/
└── combo_metadata.json
```

其中：

| 文件夹 | 内容 |
|---|---|
| `data/` | 该 combo 的全部 CSV/JSON 输出 |
| `graph/` | 该 combo 的全部 PNG 输出 |
| `logs/` | 每个 stage 的 stdout/stderr 日志 |
| `combo_metadata.json` | 当前 combo 参数、原始 run 目录、提取出的核心指标 |

---

## 10. Sweep summary 和分析

每次 sweep 完成后会生成：

```text
sweep_result/<seed_folder>/sweep_summary.csv
```

这个文件是一轮 sweep 的总成绩表。

主要列包括：

| 列名 | 含义 |
|---|---|
| `combo_folder` | 当前组合输出文件夹名 |
| `tap_num` | FIR tap 数 |
| `regularization` | LS/ridge regularization |
| `coeff_total_bits` | fixed-point 总 bit 数 |
| `coeff_frac_bits` | fixed-point 小数 bit 数 |
| `fixed_format` | 定点格式，例如 Q2.14 |
| `run_name` | 原始 pipeline run name |
| `h1_ripple_db` | 原始 H1 幅频 ripple |
| `float_dense_ripple_db` | float FIR 后 dense ripple |
| `float_dense_pass_0p1db` | float dense ripple 是否 <= 0.1 dB |
| `max_abs_coeff` | float FIR 最大系数绝对值 |
| `fixed_saturation_count` | fixed-point 系数量化 saturation 数量 |
| `fixed_dense_ripple_db` | fixed-point FIR 后 dense ripple |
| `fixed_dense_pass_0p1db` | fixed dense ripple 是否 <= 0.1 dB |
| `behavior_float_ripple_db` | multi-tone float FIR 后 ripple |
| `behavior_fixed_ripple_db` | multi-tone fixed-point FIR 后 ripple |
| `behavior_fixed_pass_0p1db` | behavior fixed ripple 是否 <= 0.1 dB |
| `qam_float_magnitude_only_evm_percent` | QAM float FIR 后 magnitude-only EVM |
| `qam_fixed_magnitude_only_evm_percent` | QAM fixed FIR 后 magnitude-only EVM |

分析 sweep：

```powershell
python sweep_test\analyze_sweep_results.py
```

如果要指定某一个 `sweep_summary.csv`：

```powershell
python sweep_test\analyze_sweep_results.py --summary-csv sweep_result\<seed_folder>\sweep_summary.csv
```

分析输出：

```text
sweep_result/<seed_folder>/sweep_analysis_report.md
sweep_result/<seed_folder>/sweep_best_combos.csv
sweep_result/<seed_folder>/sweep_group_summary.csv
sweep_result/<seed_folder>/sweep_fixed_dense_ripple_by_tap.png
sweep_result/<seed_folder>/sweep_behavior_ripple_by_tap.png
sweep_result/<seed_folder>/sweep_qam_evm_by_tap.png
sweep_result/<seed_folder>/sweep_saturation_and_coeff_range.png
```

这些文件用于报告中的参数 tradeoff 分析。

---

## 11. Dense ripple、behavior ripple、QAM EVM 的区别

### 11.1 dense ripple

dense ripple 直接在 H1/H2 的频域响应上计算，使用 H1 的 1001 个频率点。

它不依赖具体输入信号。

它回答的问题是：

```text
从频响本身看，H1 * H2_fixed 补偿后最坏 ripple 是多少？
```

这是 L1-08 最严格、最直接的主指标。

### 11.2 behavior ripple

behavior ripple 来自 multi-tone 时域仿真。

它依赖当前生成的 51 个 tone：

```text
input multi-tone
-> H1
-> FIR
-> 测每个 tone 的幅度
```

它回答的问题是：

```text
对当前 multi-tone 输入，输出 tone 的幅度是否变平？
```

它更贴近行为级仿真，但可能漏掉 dense grid 上没有被 tone 采到的最坏点。

### 11.3 QAM EVM

QAM/EVM 是辅助指标。

它更像宽带调制信号，但当前不是完整通信接收机链路。

对 L1-08 来说，应该重点看：

```text
magnitude-only EVM
```

而不是只看 full EVM。

---

## 12. 当前已有 sweep 结果

当前 `sweep_result` 里已经保存了三轮 seed sweep，每轮 27 个 combo。

已有 seed folder：

```text
sweep_result/h1_1366789340_behavior_1224877734_qam_1756070480/
sweep_result/h1_286476255_behavior_1813889189_qam_417471910/
sweep_result/h1_533121161_behavior_1688039438_qam_254967343/
```

每个 seed folder 里都有：

```text
sweep_summary.csv
sweep_analysis_report.md
sweep_best_combos.csv
sweep_group_summary.csv
4 张 sweep 分析图
27 个 combo folder
```

这些结果已经被当前 `L1-08_algorithm_design_report.md` 引用。

---

## 13. 推荐的日常工作流

### 13.1 单次验证某个参数配置

1. 修改 `L1_08_experiment_config.json` 的 `active` 参数。
2. 运行：

```powershell
python L1-08_sim\run_all_pipeline.py
```

3. 查看：

```text
L1-08_sim/data/<run_name>/run_summary.json
graph/<run_name>/*.png
```

### 13.2 扫描多个参数组合

1. 修改 `sweep_test_config.json`。
2. 运行 dry run 检查 combo：

```powershell
python sweep_test\run_sweep.py --dry-run
```

3. 正式运行：

```powershell
python sweep_test\run_sweep.py
```

4. 分析：

```powershell
python sweep_test\analyze_sweep_results.py
```

5. 查看：

```text
sweep_result/<seed_folder>/sweep_analysis_report.md
sweep_result/<seed_folder>/sweep_summary.csv
sweep_result/<seed_folder>/*.png
```

### 13.3 更换随机 seed 后重新 sweep

1. 修改 `L1_08_experiment_config.json`：

```text
active.h1.seed
active.behavior.seed
active.qam_evm.seed
```

2. 运行：

```powershell
python sweep_test\run_sweep.py
python sweep_test\analyze_sweep_results.py
```

因为 `sweep_test_config.json` 中：

```json
"group_by_current_seed": true
```

所以新的 seed 会自动生成新的 `sweep_result/<seed_folder>`。

## 14. 常见注意事项

1. 不要把 `full EVM` 直接等同于 L1-08 交付物中的 `EVM_LIN`。
2. L1-08 主目标是 magnitude ripple，不是 phase/group delay。
3. `dense ripple` 是直接看频响，通常比 multi-tone behavior 更严格。
4. `behavior ripple` 只看当前 tone 分布，可能漏掉最坏频点。
5. fixed-point 不只看 ripple，也要看 `saturation_count`。
6. seed 的数字大小不代表随机程度，随机复杂度由 `h1_random_model` 的参数范围决定。
7. sweep test 会复制每个 combo 的 `data/graph/logs` 到 `sweep_result`，用于长期保存。
8. 如果 `cleanup_sim_outputs_after_copy` 是 `false`，sweep 运行时也会在 `L1-08_sim/data` 和 `L1-08_sim/results` 留下中间 run folder。

---

## 15. 最短命令清单

单次完整运行：

```powershell
python L1-08_sim\run_all_pipeline.py
```

单次运行但跳过 QAM/EVM：

```powershell
python L1-08_sim\run_all_pipeline.py --skip-qam-evm
```

sweep dry run：

```powershell
python sweep_test\run_sweep.py --dry-run
```

正式 sweep：

```powershell
python sweep_test\run_sweep.py
```

分析当前 seed sweep：

```powershell
python sweep_test\analyze_sweep_results.py
```

分析指定 summary：

```powershell
python sweep_test\analyze_sweep_results.py --summary-csv sweep_result\<seed_folder>\sweep_summary.csv
```
