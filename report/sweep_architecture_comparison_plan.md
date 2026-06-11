# Sweep Test 架构对比改造计划（Base Plan vs Plan B）

> 本文回应导师评论：在「线性相位 FIR + 全通 IIR」（Base Plan）与「单个复系数 FIR 全覆盖」（Plan B）等多种补偿架构之间，做**定性 + 定量横向对比**，对比维度为 **① 延迟性 ② 逻辑资源消耗 ③ 稳定性 ④ 补偿效果**。
> 目标：把两个 sweep test 从「各自找最优参数」升级为「在四个维度上产出**可对齐、可叠成 trade-off 曲线**的指标」，最终自动生成导师要的架构对比表。
> 约束：**不改 pipeline 算法 / stage 划分 / 三份运行配置语义**，只改 sweep 的扫描维度、指标采集、分析程序，并新增跨 plan 对比。

---

## 0. 导师诉求拆解

| 架构 | 对应 plan | 结构 | 状态 |
|---|---|---|---|
| 线性相位 FIR + 全通 IIR | **Base Plan**（L1-08 + L1-09） | FIR 修幅度，全通 IIR 修相位/群延迟，两级级联 | 已实现 |
| 单个复系数 FIR 全覆盖 | **Plan B** | 一级复 FIR 同时修幅度 + 相位 | 已实现 |
| 单个非线性 FIR / 记忆多项式 等 | Plan C（`experimental/`） | 非线性 | 未来，预留接口 |

**四个对比维度** = 本计划所有指标必须落到的坐标轴：
1. **延迟性**：链路总群延迟（采样点 / ns），含分解项。
2. **逻辑资源消耗**：实数乘法器数、状态寄存器/延迟线、系数位宽。
3. **稳定性**：IIR 极点半径 / 稳定裕度 / 是否稳定；系数饱和数。
4. **补偿效果**：全链路 EVM、EVM_LIN（幅/相/合）、幅度 ripple、群延迟 ripple、是否达 0.1 dB。

---

## 1. 现状诊断（为什么现在无法对比）

| 维度 | Base Plan sweep 现状 | Plan B sweep 现状 |
|---|---|---|
| **输入扫描** | 5 bandwidth × 3 seed × L1-08(tap/reg/fp) × L1-09(sections/fp) = **810 组** | ❌ **只有 1 个输入**（仅扫 design 参数，不扫 profile/seed） |
| **延迟性** | ❌ 完全没采集 | ⚠️ 有 group delay ripple / fitted_delay，未汇总成「总延迟」 |
| **逻辑资源** | ❌ 完全没有 | ✅ `estimated_real_multiplier_count`（= 4 × tap） |
| **稳定性** | ⚠️ CSV **已有** `l1_09_max_pole_radius`/`stable`/饱和数，但**分析程序未读** | ✅ 饱和数（FIR 结构天然稳定） |
| **补偿效果** | ⚠️ 分析程序只看 `qam_fixed_magnitude_only_evm`（**仅 L1-08 幅度**），丢了 L1-09 相位补偿这一半 | ✅ 全 EVM + EVM_LIN(幅/相/合) + 幅度 ripple |

### 两个致命问题
1. **Plan B 不扫输入** → 两个架构没有在同一组 H1 / 带宽 / seed 上评估，对比在统计上不成立。
2. **Base 分析程序只评估了架构的一半**：
   - 头部「补偿效果」用的是 `qam_fixed_magnitude_only_evm_percent`（L1-08 幅度），**未计入 L1-09 的相位补偿**；
   - 已躺在 `sweep_summary.csv` 里的 L1-09 稳定性、全链路 EVM、EVM_LIN 被 `analyze_sweep_results.py` 的 `SweepRow` 完全忽略。

### 关键代码位置（事实依据）
- Base runner 已采集 L1-09：`L1-08+L1-09_sim_base_plan/sweep_test/existing_pipeline_runner.py` `_extract_metrics`（约 255–263 行：`l1_09_max_pole_radius` / `l1_09_qam_fixed_evm_percent` / `l1_09_evm_lin_fixed_metrics`）。
- Base 分析器忽略它们：`analyze_sweep_results.py` 的 `SweepRow`（约 26–51 行）无任何 L1-09 / 延迟 / 资源字段。
- Base 完全没有延迟、资源指标（runner、analyzer 均无）。
- Plan B 资源模型：`estimated_real_multiplier_count`（复 FIR 非线性相位 → 不可折叠 → 4 × tap）。
- Plan B 不扫 profile/seed：`run_plan_b_sweep.py` 的 cases 只对 (fs, tap, reg, delay, fixed_point) 做笛卡尔积。

---

## 2. 统一指标词表（让两个 plan 可比的契约）

**两个 sweep 的 `sweep_summary.csv` 都必须输出同名、同单位的下列列。** 这是所有改动的地基。

| 维度 | 统一列名 | Base Plan 来源 | Plan B 来源 |
|---|---|---|---|
| 延迟 | `total_latency_samples` / `total_latency_ns` | FIR `group_delay_samples` + IIR `target_delay_ns`（mean group delay） | 复 FIR `reference_delay_samples` |
| 延迟（分解） | `fir_bulk_delay_samples`、`iir_added_delay_samples` | H2 fir + allpass | `reference_delay_samples` / 0 |
| 资源 | `real_multiplier_count` | FIR(对称折叠) + IIR(节数×每节×复2) | 4 × tap |
| 资源 | `state_reg_count`、`coeff_total_bits` | FIR 延迟线 + IIR 状态 | FIR 延迟线 |
| 稳定性 | `max_pole_radius`、`pole_margin`(=1−r)、`is_stable` | L1-09 已有 | FIR 恒稳 → r=0、margin=1、True |
| 稳定性 | `saturation_count` | L1-08 + L1-09 饱和合计 | 复 FIR 饱和 |
| 补偿效果 | `fixed_full_evm_percent` | **全链路** `l1_09_qam_fixed_evm_percent` | `after_plan_b_fixed_evm_percent` |
| 补偿效果 | `fixed_evm_lin_percent`、`magnitude_ripple_db`、`group_delay_ripple_ns`、`pass_0p1db` | L1-09 EVM_LIN + behavior ripple | Plan B EVM_LIN + 设计 ripple |

> **资源模型必须抽到 `shared_sim/resource_model.py`，两个 sweep 共用同一套乘法器/存储计数**，否则「逻辑资源消耗」这一维度的对比不公平、不可信。

### 量级预估（代入真实跑出的参数，验证 trade-off 真实存在）
- **Base**：96-tap 实 FIR（对称折叠 ≈ 96）+ 8 节复全通（≈ 32）≈ **~130 实乘法器**；**有 IIR 稳定性风险**；总延迟 ≈ FIR + IIR 附加 ≈ **~12 ns**。
- **Plan B**：256-tap 复 FIR = **1024 实乘法器**（非线性相位不可折叠）；**恒稳**；延迟 ≈ **~10.6 ns**。
- → 这正是导师要的结论形态：**Base 省资源但两级、有稳定性风险；Plan B 资源贵但单级、恒稳**。
- ⚠️ 每节全通的乘法器数取决于 RTL 结构（1-mult / 2-mult lattice），需在资源模型里**显式选定一种约定**并在报告注明。

---

## 3. 怎么 sweep（扫描维度设计）

### 3.1 Base Plan —— 扫描维度基本正确，补采集即可
- 保留 `bandwidth_profiles × seed_cases`（输入鲁棒性，导师看重）。
- 旋钮与四维度的语义映射：
  - `l1_08_tap_num`：幅度补偿↑ / 资源↑ / 延迟↑
  - `l1_09_allpass_sections`：相位补偿↑ / **稳定性风险↑** / 资源↑ / 附加延迟
  - 两级 `fixed_point`：资源（位宽）↔ 稳定性（极点）/ 精度
- 810 组偏大：可先固定 `regularization`（已是单值），`fixed_point` 收敛到 2 档跑全量；需要细扫时再放开。

### 3.2 Plan B —— 必须补输入扫描（**本计划最重要的改动**）
- `config_plan_b_sweep.json` 增加与 Base **同名同值的** `bandwidth_profiles` + `seed_cases`。
- `run_plan_b_sweep.py` 外层加 `(profile, seed)` 循环：每个组合**先生成对应 H1**（复用已加的 H1 自动生成逻辑），再扫 design 参数。
- tap 范围**不必**与 Base 相等（Plan B 单级要同时修幅+相，本就需要更多 tap）；公平性由「统一延迟/资源指标」保证，而非 tap 数相等。

### 3.3 配置形态（目标）
```jsonc
// config_plan_b_sweep.json 增补
{
  "bandwidth_profiles": ["bw_500m","bw_1g","bw_2g","bw_4g","bw_8g"],
  "seed_cases": [ /* 与 config_base_plan_sweep.json 完全一致的 3 组 */ ],
  "design_sweep": { "tap_num": [128,192,256,320], "regularization": [1e-6,1e-5,1e-4] },
  "fixed_point_sweep": { "choices": [ {"coeff_total_bits":16,"coeff_frac_bits":13}, {"coeff_total_bits":18,"coeff_frac_bits":15} ] }
}
```

---

## 4. 怎么改 analysis program

### 4.1 Base 侧
| 文件 | 改动 |
|---|---|
| `existing_pipeline_runner.py` `_extract_metrics` | 【改】补采**延迟**（FIR `group_delay_samples`、IIR `target_delay_ns`/`group_delay_mean_ns`）和**资源**（调用 `shared_sim/resource_model.py`）；输出统一词表列 |
| `analyze_sweep_results.py` | 【改】`SweepRow` 增加 L1-09 + 延迟 + 资源字段；新增四段分析：**全链路 EVM**（替换当前「仅 L1-08 幅度」口径）、**稳定性**（pole radius vs sections/格式、统计不稳定组合数）、**延迟**、**资源-效果 Pareto**（借鉴 Plan B profiler） |

### 4.2 Plan B 侧
| 文件 | 改动 |
|---|---|
| `analyze_plan_b_sweep_results.py` | 【改】① 输出统一词表列名；② 增加按 `profile / seed` 分组（现在完全没有） |
| `sweep_result_profiler.py` | 复用：其 Pareto / 复合评分机制移植到 Base，统一两侧 profiler |

### 4.3 新增：跨 plan 对比（导师那张表的落地）
| 文件 | 作用 |
|---|---|
| `compare_plans.py`（建议放 `report/` 或根） | 【新】读两个 sweep 的 `sweep_summary.csv`，按 bandwidth profile 对齐，产出**架构对比报告**：<br>· 「达到同一 EVM 目标」时比**资源 + 延迟**<br>· 「同等资源」时比**补偿效果**<br>· 标注**稳定性裕度**<br>· schema 预留 Plan C / 非线性 FIR 接入位 |

对比报告核心表（示意）：

| Profile | 架构 | 全 EVM(%) | 幅度 ripple(dB) | 总延迟(ns) | 实乘法器 | 稳定裕度 | 达标 |
|---|---|---:|---:|---:|---:|---:|:--:|
| bw_2g | Base (FIR+APIIR) | … | … | ~12 | ~130 | 1−r | ✓/✗ |
| bw_2g | Plan B (complex FIR) | … | … | ~10.6 | 1024 | 1.0(恒稳) | ✓/✗ |

---

## 5. 落地优先级（信息回收 / 改动量排序）

1. **统一资源模型** `shared_sim/resource_model.py`（公平性地基）。
2. **Base 分析器读全 L1-09 + 全链路 EVM**（最小改动、最大信息回收——数据已在 CSV）。
3. **Base 补采延迟 / 资源**（runner `_extract_metrics`）。
4. **Plan B sweep 加 profile / seed 扫描**（`config_plan_b_sweep.json` + `run_plan_b_sweep.py`）。
5. **跨 plan 对比报告** `compare_plans.py`。

> 建议先做 1 + 2，立即把"架构的另一半"信息救回来；再做 3 + 4 打平两侧；最后 5 出导师要的对比表。

---

## 6. 待确认问题
1. 资源模型乘法器约定：全通每节按 **2-mult** 还是 **1-mult lattice**？是否同时统计加法器 / 存储？
2. 「补偿效果」主指标用**全 QAM EVM** 还是 **EVM_LIN**？（建议两者都报，主排序用全 QAM EVM。）
3. Plan B 是否跑满 5 × 3 = 15 个输入 × 现有 design 网格（组合数会显著增大），还是先取子集？
4. 跨 plan「对齐口径」用「同 EVM 目标比成本」还是「同成本比效果」为主？（建议同 EVM 目标为主。）
