# Sweep 主扫 A 执行计划（bandwidth = bw_1g）

> 本文是交给 composer 独立执行的完整计划，**完全取代本文件旧内容**。
> composer 看不到聊天上下文，所有必要信息都写在本文内。请严格按任务顺序执行，每个任务都有「文件」「现状」「目标」「精确改动」「验收」。

---

## 0. 背景（必读，理解 why）

本项目对比两种均衡补偿架构，在同一组输入上评估**补偿效果**与**稳定性**两个维度（延迟维度本轮不扫，固定旋钮）：

| 架构 | 目录 | 结构 | 稳定性 |
|---|---|---|---|
| **Base Plan** | `L1-08+L1-09_sim_base_plan/` | L1-08 实线性相位 FIR（修幅度）+ L1-09 全通 IIR（修相位/群延迟），两级级联 | IIR 有极点，**可能失稳** |
| **Plan B** | `L1-08+L1-09_sim_planB/` | 单级复系数 FIR，同时修幅度+相位 | FIR 无反馈，**恒稳** |

**本轮目标**：执行「主扫 A」——只扫 **bandwidth profile = `bw_1g`**，保留 **3 个 seed_cases** 做鲁棒性 ensemble，让两个 plan 在**同一组 H1**（同 profile 同 seed → H1 完全相同）上，产出可对齐的补偿效果 + 稳定性指标。

**关键约束（不可违反）**：
1. **不改 pipeline 算法 / stage 划分 / 三份运行配置**（`config_input.json`、`config_base_plan.json`、`config_plan_b.json`）的语义。
2. **不动延迟旋钮**：Base 的 `l1_09.allpass.margin_ns` 保持配置默认（`null`=自动），Plan B 的 `reference_delay_samples` 保持 `null`（=自动 `(tap-1)/2`）。延迟专项是另一轮，本轮不碰。
3. `shared_sim/` 位置不变（它是 repo-root 锚点）。
4. 两个 plan 必须用**完全相同的 `seed_cases`**（同 name、同 h1_seed/behavior_seed/qam_seed），否则 H1 不一致、跨架构对比失效。

---

## 1. 任务总览（按顺序执行）

| # | 任务 | 主要文件 |
|---|---|---|
| 1 | 改 Base sweep 参数 | `config_base_plan_sweep.json` |
| 2 | 改 Plan B sweep 参数 | `config_plan_b_sweep.json` |
| 3 | 给 Plan B sweep 程序加 profile/seed ensemble 支持 | `L1-08+L1-09_sim_planB/sweep_test/run_plan_b_sweep.py` |
| 4 | 改造 Base 分析程序（读 L1-09 + 全链路 EVM + 稳定性） | `L1-08+L1-09_sim_base_plan/sweep_test/analyze_sweep_results.py` |
| 5 | 改造 Plan B 分析程序（加 profile/seed 分组） | `L1-08+L1-09_sim_planB/sweep_test/analyze_plan_b_sweep_results.py` |
| 6 | 跑通并验证 | 见 §7 |

---

## 2. 任务 1：Base sweep 参数

**文件**：`config_base_plan_sweep.json`

**现状**：`sweep.bandwidth_profiles` = 5 个；`l1_08.tap_num`=[64,80,96]；`l1_08.fixed_point`=3 档；`l1_09.fixed_point`=[Q3.15,Q4.14]。

**目标参数**：只改 `sweep` 块内的取值，`paths`/`output`/`stages` 块**保持不变**。

```json
"sweep": {
  "description": "Main sweep A: bw_1g only, 3 seeds, focus on full-chain compensation + L1-09 stability boundary.",
  "bandwidth_profiles": ["bw_1g"],
  "seed_cases": [
    { "name": "seed_a", "h1_seed": 1040330957, "behavior_seed": 1068822328, "qam_seed": 225239571 },
    { "name": "seed_b", "h1_seed": 297216821,  "behavior_seed": 953625061,  "qam_seed": 453249928 },
    { "name": "seed_c", "h1_seed": 1571318759, "behavior_seed": 1925817879, "qam_seed": 140927704 }
  ],
  "l1_08": {
    "tap_num": [80, 96],
    "regularization": [0.0001],
    "fixed_point": [
      { "name": "Q3.13", "coeff_total_bits": 16, "coeff_frac_bits": 13 }
    ]
  },
  "l1_09": {
    "allpass_sections": [6, 8, 10],
    "fixed_point": [
      { "name": "Q3.15", "coeff_total_bits": 18, "coeff_frac_bits": 15 },
      { "name": "Q4.14", "coeff_total_bits": 18, "coeff_frac_bits": 14 },
      { "name": "Q5.13", "coeff_total_bits": 18, "coeff_frac_bits": 13 }
    ]
  }
}
```

**说明**：
- `l1_08.tap_num=[80,96]`、`l1_08.fixed_point` 固定一档 `Q3.13`、`l1_08.regularization` 固定 `1e-4` → 聚焦 L1-09。
- `l1_09.fixed_point` 新增 **`Q5.13`（18bit/13frac，故意更粗）**：用于把某些 `(allpass_sections, fixed_point)` 组合推向**失稳**，从而扫出稳定性边界。`allpass_sections=10` + `Q5.13` 最可能失稳。
- **组合数 = 1(profile) × 3(seed) × 2(tap) × 1(reg) × 1(l1_08 fp) × 3(sections) × 3(l1_09 fp) = 54 组**。

**验收**：JSON 合法；`sweep.bandwidth_profiles` 只有 `bw_1g`；`l1_09.fixed_point` 含 3 档含 `Q5.13`。

---

## 3. 任务 2：Plan B sweep 参数

**文件**：`config_plan_b_sweep.json`

**现状**：有 `paths`/`input`/`output`/`design_sweep`/`fixed_point_sweep`，但**没有 bandwidth/seed 概念**。文件尾部有几行垃圾字符（如 `、`），一并清掉。

**目标**：新增 `sweep` 块（与 Base 同名同值的 `bandwidth_profiles` + `seed_cases`），其余保留。完整文件：

```json
{
  "sweep_name": "plan_b_complex_fir_sweep_test",
  "description": "Plan B main sweep A: bw_1g only, 3 seeds (same as Base), sweep complex FIR design + fixed-point. Input from config_input.json.",
  "paths": {
    "repo_root": ".",
    "input_config": "config_input.json",
    "plan_b_config": "config_plan_b.json",
    "output_root": "sweep_result"
  },
  "sweep": {
    "bandwidth_profiles": ["bw_1g"],
    "seed_cases": [
      { "name": "seed_a", "h1_seed": 1040330957, "behavior_seed": 1068822328, "qam_seed": 225239571 },
      { "name": "seed_b", "h1_seed": 297216821,  "behavior_seed": 953625061,  "qam_seed": 453249928 },
      { "name": "seed_c", "h1_seed": 1571318759, "behavior_seed": 1925817879, "qam_seed": 140927704 }
    ]
  },
  "input": {
    "run_dir": null,
    "h1_csv": null
  },
  "output": {
    "sweep_result_root": "sweep_result",
    "sweep_folder_name": null,
    "save_case_outputs": true,
    "save_case_graphs": true
  },
  "design_sweep": {
    "fs_hz": 12000000000.0,
    "tap_num": [192, 256, 320],
    "reference_delay_samples": [null],
    "regularization": [0.000001, 0.00001]
  },
  "fixed_point_sweep": {
    "choices": [
      { "coeff_total_bits": 16, "coeff_frac_bits": 13 },
      { "coeff_total_bits": 18, "coeff_frac_bits": 15 }
    ]
  }
}
```

**说明**：
- `seed_cases` **必须与任务 1 完全相同**（保证 H1 一致）。
- **组合数 = 1(profile) × 3(seed) × 3(tap) × 2(reg) × 1(delay) × 2(fp) = 36 组**。

**验收**：JSON 合法、无尾部垃圾字符；`sweep.seed_cases` 与 `config_base_plan_sweep.json` 三个 seed 完全一致。

---

## 4. 任务 3：Plan B sweep 加 profile/seed ensemble 支持

**文件**：`L1-08+L1-09_sim_planB/sweep_test/run_plan_b_sweep.py`

### 4.1 现状（必读）
- `main()` 只生成**一条** H1（`resolve_sweep_run_dir` → `generate_h1_run`），然后在其上对 `(fs, tap, reg, delay, fixed_point)` 做笛卡尔积扫描，写一个 `sweep_summary.csv`。
- 完全没有 profile/seed 维度。
- H1 由子进程 `shared_sim/h1_source.py` 生成，**通过环境变量控制 profile 和 seed**。

### 4.2 H1/behavior/QAM 的 seed 注入机制（关键，照此实现）
`shared_sim/config.py` 定义了这些环境变量，下游自动读取：
- `L1_08_PROFILE` → 选 bandwidth profile
- `L1_08_SEED_CASE` → seed case 名（仅用于标注/run_summary）
- `L1_08_H1_SEED` → 覆盖 H1 随机种子
- `L1_08_BEHAVIOR_SEED` → 覆盖 behavior 种子
- `L1_08_QAM_SEED` → 覆盖 QAM 种子

机制：`config.py` 的 `_apply_input_seed_env_overrides()` 会把 `L1_08_H1_SEED`/`L1_08_BEHAVIOR_SEED`/`L1_08_QAM_SEED` 覆盖进 input config 的对应 `seed` 字段；`h1_source.py` 生成 H1 时读取被覆盖后的 h1 seed。**因此：设置好这 5 个 env 后，子进程生成的 H1 与 Base 在同 profile/同 seed 下完全一致。**

⚠️ 注意：Plan B 的 QAM 验证（`run_plan_b_qam_evm_validation`）是**在 sweep 进程内**调用的（不是子进程）。`QamEvmConfig.seed` 在 `parse_args()` 时从 config 读默认值，**不会**自动随 env 变化。所以**必须在每个 ensemble 成员里，用该 seed_case 的 `qam_seed` 显式构造 `QamEvmConfig`**（见 4.4 第 4 步）。

### 4.3 改动目标
把 `main()` 重构为**双层循环**：外层遍历 `(profile, seed_case)` ensemble，内层遍历现有 design × fixed_point。所有结果写入**一个**合并 `sweep_summary.csv`，每行带 `profile/seed_case/h1_seed/behavior_seed/qam_seed` 标注。保留无 ensemble 时的旧单 H1 行为（向后兼容）。

### 4.4 精确改动

**(1) 读取 ensemble 配置**：解析 `config_plan_b_sweep.json` 的 `sweep` 块：
- `bandwidth_profiles`（list，缺省 `[None]`）
- `seed_cases`（list of dict，每个含 `name/h1_seed/behavior_seed/qam_seed`；缺省 `[None]`）
- 若两者都为空/缺省 → **走旧单 H1 逻辑**（兼容）。

**(2) 新增按 seed_case 设置 env 的函数**（在现有 `sweep_env()` 基础上扩展）：
```python
def ensemble_env(base_env: dict[str, str], profile: str | None, seed_case: dict | None) -> dict[str, str]:
    env = dict(base_env)
    if profile:
        env["L1_08_PROFILE"] = profile
    if seed_case is not None:
        env["L1_08_SEED_CASE"]     = str(seed_case["name"])
        env["L1_08_H1_SEED"]       = str(seed_case["h1_seed"])
        env["L1_08_BEHAVIOR_SEED"] = str(seed_case["behavior_seed"])
        env["L1_08_QAM_SEED"]      = str(seed_case["qam_seed"])
    return env
```

**(3) 外层 ensemble 循环**：对每个 `(profile, seed_case)`：
- `env_i = ensemble_env(sweep_env(), profile, seed_case)`
- 同时把这些 env **设进 `os.environ`**（因为下游 in-process 读取 input config 时依赖 `os.environ`），生成完后可不还原（每个成员重设即可）。
- 生成该成员的 H1：复用 `generate_h1_run(env_i)`（子进程已 inherit env_i）→ 得到 `run_dir_i`。
- 该成员的 QAM seed = `seed_case["qam_seed"]`（若 seed_case 为 None 用 args.seed）。

**(4) 内层 design 循环**：在 `run_dir_i` 上对 `(fs, tap, reg, delay, fixed_point)` 扫描（沿用现有 `run_plan_b_case` + `run_plan_b_qam_evm_validation` + `run_evm_lin_from_total_responses` 流程），但：
- `QamEvmConfig(...)` 的 `seed=` 用**该成员的 qam_seed**，不要用 parse 时的全局默认。
- `case_id` 前缀加 profile+seed，避免不同成员的同名 case 目录冲突：
  ```python
  member_prefix = f"{profile or 'active'}_{(seed_case or {}).get('name','active')}_"
  this_case_id = member_prefix + case_id(tap, reg, ctb, cfb)
  ```

**(5) CSV 增列**：在 `sweep_fieldnames()` 开头（`case_id` 之后）加入：
```
"profile", "seed_case", "h1_seed", "behavior_seed", "qam_seed"
```
并在每行 row dict 写入对应值（成员级，所有 design case 共享该成员的 profile/seed）。

**(6) 输出目录**：全部 ensemble × design 结果写入**同一个** sweep 文件夹下的单个 `sweep_summary.csv`。文件夹名建议 `sweep_result/plan_b_qam_sweep_bw1g_3seed`（或沿用 `resolve_configured_output_dir`，但确保多成员不互相覆盖——单一 summary、各 case 子目录用带前缀的 case_id 区分）。`parameter_setting_comb.json` 记录 ensemble + design 全部组合。

### 4.5 验收
- 零参数运行 `python L1-08+L1-09_sim_planB\sweep_test\run_plan_b_sweep.py` 跑完 **36 行**（3 seed × 12 design），`status=ok`。
- `sweep_summary.csv` 含 `profile/seed_case/h1_seed/behavior_seed/qam_seed` 列，且 3 个 seed 的 `h1_seed` 与 `config_base_plan_sweep.json` 一致。
- 不传 ensemble（旧 config）时仍能单 H1 跑通（兼容未破坏）。

---

## 5. 任务 4：Base 分析程序改造

**文件**：`L1-08+L1-09_sim_base_plan/sweep_test/analyze_sweep_results.py`

### 5.1 现状（必读）
- `SweepRow` **只含 L1-08 字段**，补偿效果主指标用 `qam_fixed_magnitude_only_evm_percent`（**只算了 L1-08 幅度补偿那一半**）。
- L1-09 的稳定性、全链路 EVM **已经在 `sweep_summary.csv` 里**（由 `existing_pipeline_runner.py` 的 `_extract_metrics` 写入），但分析器**完全没读**。
- 现有 bandwidth_vs_* 三张图在单 bandwidth 下会退化。

### 5.2 CSV 中已存在、需要读入的 L1-09 列（无需改 runner，列已存在）
```
l1_09_fixed_saturation_count
l1_09_fixed_stable                 (字符串 true/false)
l1_09_max_pole_radius
l1_09_qam_float_evm_percent
l1_09_qam_fixed_evm_percent                          <- 全链路 QAM EVM（fixed），补偿效果主指标
l1_09_qam_float_magnitude_only_evm_percent
l1_09_qam_fixed_magnitude_only_evm_percent
l1_09_evm_lin_float_metrics        (JSON 字符串)
l1_09_evm_lin_fixed_metrics        (JSON 字符串)
l1_09_allpass_sections             (combo 列)
l1_09_fixed_format                 (combo 列)
```

### 5.3 精确改动
**(1) `SweepRow` 增字段**：加入 `allpass_sections:int`、`l1_09_fixed_format:str`、`l1_09_max_pole_radius:float`、`l1_09_fixed_stable:bool`、`l1_09_fixed_saturation_count:int`、`l1_09_qam_fixed_evm_percent:float`、`l1_09_qam_fixed_magnitude_only_evm_percent:float`。`load_summary` 里解析这些列（`l1_09_fixed_stable` 用现有 `parse_bool`）。

**(2) 补偿效果主指标改口径**：分析与报告里，把「全链路补偿效果」主指标从 `qam_fixed_magnitude_only_evm_percent`（仅 L1-08）改为 **`l1_09_qam_fixed_evm_percent`（after L1-08+L1-09 全链路 QAM EVM）**。保留 L1-08 列做对照，但**报告的「最优/排序」以全链路为准**。

**(3) 新增「稳定性」分析段**（核心，对应导师维度）：
- 按 `(allpass_sections × l1_09_fixed_format)` 制表：每格的 `max_pole_radius`（min/mean/max）、`is_stable` 计数、`saturation_count`。
- 统计**失稳组合数**（`l1_09_fixed_stable=false` 或 `max_pole_radius>=1`），列出失稳的 `(sections, format, seed)`。
- 画一张稳定性图：x=allpass_sections，y=max_pole_radius，按 l1_09_fixed_format 分色，画 `radius=1` 危险线。

**(4) 单 bandwidth 适配**：bandwidth 只有 `bw_1g` → **删除或跳过** `bandwidth_vs_*` 三张退化图与「Bandwidth Sweep Result」段；改为强调：
- 按 **seed** 分组：全链路 EVM 的 best/mean/worst（鲁棒性）。
- 按 **tap_num / allpass_sections / l1_09_fixed_format** 分组的全链路 EVM 与稳定性。

**(5) 图表更新**：`plot_metric_by_tap` 系列把主指标换成 `l1_09_qam_fixed_evm_percent`；新增稳定性图（见 (3)）。保留饱和/系数图。

**(6) 报告章节**（建议结构）：1.Scope（注明 bw_1g、3 seed、54 组、口径=全链路）→ 2.Overall（达标计数、失稳计数）→ 3.Best Combos（全链路 EVM 排序）→ 4.Stability（新增）→ 5.Seed Robustness → 6.Group Summary（tap/sections/l1_09 format）→ 7.Interpretation。

### 5.4 验收
- 对任务 1 产出的 54 行 CSV，`analyze_sweep_results.py` 零参数跑通，生成 `sweep_analysis_report.md` + 稳定性图。
- 报告「最优组合」「补偿效果」均基于 `l1_09_qam_fixed_evm_percent`（全链路），不再是 L1-08-only。
- 报告含「Stability」段，能列出失稳组合（若 `Q5.13`+高 sections 触发）。

---

## 6. 任务 5：Plan B 分析程序改造

**文件**：`L1-08+L1-09_sim_planB/sweep_test/analyze_plan_b_sweep_results.py`

### 6.1 现状
指标已很全（QAM EVM、EVM_LIN 幅/相/合、ripple、群延迟、相位误差、饱和），但**完全没有 profile/seed 分组**（因为旧 sweep 不产 seed 列）。

### 6.2 精确改动
**(1) 读 ensemble 列**：`PlanBSweepRow` 增 `profile:str`、`seed_case:str`、`h1_seed/behavior_seed/qam_seed`（来自任务 3 新增的 CSV 列；用 `row.get(...)` 容错，缺列时填 `"active"`/None，保持对旧 CSV 兼容）。

**(2) 加 seed 鲁棒性分析**：在现有 group_summary / best 分析基础上，新增按 `seed_case` 与 `(tap × fixed_format)` 的分组：各设计参数下 `after_plan_b_fixed_evm_percent`、`fixed_total_magnitude_ripple_db` 在 3 个 seed 上的 best/mean/worst。

**(3) 稳定性**：FIR 恒稳 → 报告里单列一句「Plan B 复 FIR 无反馈，结构无条件稳定」，只汇报 `saturation_count`（按 fixed_format）。

**(4) 报告补一段「Seed Robustness」**，并在 Summary 标注 bw_1g、3 seed、36 组。

### 6.3 验收
- 对任务 3 产出的 36 行 CSV 跑通 `analyze_plan_b_sweep_results.py`，报告含 seed 分组。
- 兼容旧无 seed 列的 CSV（不崩）。

---

## 7. 任务 6：跑通并验证（composer 必须执行）

按顺序执行并确认每步成功：

```powershell
# 1) Base 主扫（54 组，约 40 分钟）
python L1-08+L1-09_sim_base_plan\sweep_test\run_sweep.py --dry-run   # 先确认 54 组
python L1-08+L1-09_sim_base_plan\sweep_test\run_sweep.py
python L1-08+L1-09_sim_base_plan\sweep_test\analyze_sweep_results.py

# 2) Plan B 主扫（36 组，约 10 分钟）
python L1-08+L1-09_sim_planB\sweep_test\run_plan_b_sweep.py
python L1-08+L1-09_sim_planB\sweep_test\analyze_plan_b_sweep_results.py
```

**总验收清单**：
1. Base dry-run 报 **54** 组；实跑 54 行全 `ok`。
2. Plan B 实跑 **36** 行全 `status=ok`，含 profile/seed 列。
3. 两个 `sweep_summary.csv` 的 3 个 seed 的 `h1_seed` 一致（→ H1 对齐）。
4. Base 报告：补偿效果口径 = 全链路 `l1_09_qam_fixed_evm_percent`；含 Stability 段；无退化的 bandwidth 图。
5. Plan B 报告：含 seed 鲁棒性分组。
6. 不破坏现有 pipeline（`run_all_pipeline.py` 等仍可单跑）。

---

## 8. 交付物清单

| 文件 | 改动类型 |
|---|---|
| `config_base_plan_sweep.json` | 改 `sweep` 取值（bw_1g、tap[80,96]、l1_09 fp 加 Q5.13） |
| `config_plan_b_sweep.json` | 加 `sweep` 块（bw_1g + 3 seed）、清尾部垃圾字符 |
| `L1-08+L1-09_sim_planB/sweep_test/run_plan_b_sweep.py` | 加 profile/seed ensemble 双层循环 + CSV 增列 |
| `L1-08+L1-09_sim_base_plan/sweep_test/analyze_sweep_results.py` | 读 L1-09、换全链路口径、加 Stability 段、适配单 bandwidth |
| `L1-08+L1-09_sim_planB/sweep_test/analyze_plan_b_sweep_results.py` | 加 profile/seed 分组 + seed 鲁棒性段 |
| `sweep_result/...` | 两份新 sweep 结果 + 分析报告（运行产出） |

---

## 9. 关键参数速查（防止执行偏差）

- bandwidth：仅 `bw_1g`。
- seed_cases（两 plan 完全相同）：`seed_a`(h1=1040330957)、`seed_b`(h1=297216821)、`seed_c`(h1=1571318759)。
- Base：`l1_08.tap_num=[80,96]`、`l1_08.reg=[1e-4]`、`l1_08.fp=[Q3.13]`、`l1_09.sections=[6,8,10]`、`l1_09.fp=[Q3.15,Q4.14,Q5.13]` → **54 组**。
- Plan B：`tap=[192,256,320]`、`reg=[1e-6,1e-5]`、`reference_delay=[null]`、`fp=[Q16.13,Q18.15]` → **36 组**。
- 延迟旋钮本轮**不扫**：`margin_ns` 与 `reference_delay_samples` 保持 `null`。
- Base 补偿主指标：`l1_09_qam_fixed_evm_percent`（全链路，已在 CSV）。
- 稳定性指标：`l1_09_max_pole_radius` / `l1_09_fixed_stable` / `saturation_count`。
