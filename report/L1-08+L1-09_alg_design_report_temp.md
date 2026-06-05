# L1-08 + L1-09 Algorithm Design Report Temp

## 0. 这份文档的目的

这份文档解释当前 repo 里 L1-08 + L1-09 仿真程序到底在做什么，以及背后的数学原理。它假设读者没有接触过幅频补偿、相位补偿、FIR、all-pass filter、fixed-point 或 EVM，所以会先从直观概念讲起，再对应到当前程序结构。

当前完整 pipeline 入口是：

```powershell
.venv\Scripts\python.exe run_full_l1_08_l1_09_pipeline.py
```

它会先运行 L1-08 幅频补偿仿真，再把 L1-08 生成的同一个 H1 phase 交给 L1-09 做 group-delay / phase 补偿仿真。

当前输出规则是：

```text
data/   保存 csv、json 等数据文件
graph/  保存 png 图像文件
```

一次完整 pipeline 会生成类似：

```text
data/full_combined_YYYYMMDD_HHMMSS/
graph/full_combined_YYYYMMDD_HHMMSS/
```

---

## 1. 整体问题：我们到底在补偿什么

真实硬件链路不是理想的。信号从前端硬件经过时，频率响应会出现两个主要问题：

1. 不同频率的增益不一样，也就是 amplitude / magnitude 不平坦。
2. 不同频率的相位延迟不一样，也就是 phase / group delay 不平坦。

当前程序把真实硬件链路抽象成一个复数频率响应：

```text
H1(f) = |H1(f)| * exp(j * phi_H1(f))
```

其中：

```text
|H1(f)|        表示幅度响应
phi_H1(f)     表示相位响应
f             表示频率
j             表示虚数单位
```

L1-08 做的是幅频补偿：

```text
目标：让 |H1(f) * H2(f)| 尽量平坦
```

L1-09 做的是相位 / group delay 补偿：

```text
目标：让 H1(f) 经过 L1-09 all-pass filter 后的 group delay 尽量接近常数
```

所以整体串联关系可以理解为：

```text
输入 I/Q 信号
    ↓
硬件链路 H1：产生幅度不平坦 + 相位不平坦
    ↓
L1-08 FIR：补偿幅度不平坦
    ↓
L1-09 all-pass IIR：补偿 group delay / phase 不平坦
    ↓
输出信号
```

---

## 2. 当前程序结构

当前 repo 里和算法最相关的目录如下：

```text
L1-08_sim/
    H1_full_combined_random_generator.py
    H2_target_generator.py
    H2_fir_designer.py
    H2_fixed_point_quantizer.py
    L1_08_behavior_sim.py
    L1_08_qam_evm_sim.py
    run_all_pipeline.py

L1_09_sim/
    L1_09_group_delay_analyzer.py
    L1_09_allpass_designer.py
    L1_09_fixed_point_quantizer.py
    L1_09_evm_lin_calculator.py
    L1_09_qam_evm_validator.py
    run_all_l1_09_pipeline.py

magnitude/
    H1_slope_random_generator.py
    H1_ripple_random_generator.py
    H1_notch_bump_random_generator.py
    H1_edge_rolloff_random_generator.py
    H1_measurement_noise_random_generator.py

phase/
    H1_linear_phase_delay_random_generator.py
    H1_phase_ripple_random_generator.py
    H1_local_phase_distortion_random_generator.py
    H1_group_delay_ripple_random_generator.py
    H1_phase_noise_random_generator.py

run_full_l1_08_l1_09_pipeline.py
```

`run_full_l1_08_l1_09_pipeline.py` 是最外层总入口。它做两件事：

1. 调用 `L1-08_sim/run_all_pipeline.py`，完成 H1 生成、H2 设计、fixed-point、behavior、QAM。
2. 找到刚刚生成的 `data/full_combined_...` run folder，再调用 `L1_09_sim/run_all_l1_09_pipeline.py`，完成 group delay 分析、all-pass 设计、fixed-point、EVM 验证。

---

## 3. H1 object 和输入数据模型

当前程序的核心数据对象是 `H1`，定义在：

```text
L1-08_sim/H1_common.py
```

它包含：

```text
name       这个 H1 feature 的名字
freq_hz    频率数组，单位 Hz
h_db       幅度响应，单位 dB
phase_rad  相位响应，单位 rad
```

也就是说，一个 H1 object 描述的是一条复数频率响应曲线：

```text
H1(f_k) = 10^(h_db[k] / 20) * exp(j * phase_rad[k])
```

为什么 `h_db` 转线性幅度要用 `10^(dB/20)`？

因为 dB 对幅度的定义是：

```text
h_db = 20 * log10(|H|)
```

所以反过来：

```text
|H| = 10^(h_db / 20)
```

当前程序不是只生成一条简单曲线，而是把多个现实硬件可能出现的 feature 叠加起来：

```text
H1 magnitude = slope + ripple + notch/bump + edge_rolloff + measurement_noise
H1 phase     = linear_delay + phase_ripple + local_phase_distortion + group_delay_ripple + phase_noise
```

最后得到：

```text
magnitude_combined.csv
phase_combined.csv
together.csv
```

其中 `together.csv` 是后续 L1-08 和 L1-09 都会读取的总 H1 数据。

---

## 4. H1 magnitude random generation

### 4.1 Slope

`H1_slope_random_generator.py` 模拟通带内整体向上或向下倾斜的增益误差。

直观理解：

```text
低频增益略高，高频增益略低
或
低频增益略低，高频增益略高
```

数学上可以近似写成：

```text
x = (f - f_min) / (f_max - f_min)
h_slope_db(f) = a * (x - 0.5) + b
```

其中：

```text
x       归一化频率，范围 0 到 1
a       slope 强度，决定峰峰值变化
b       整体 offset
```

### 4.2 Ripple

`H1_ripple_random_generator.py` 模拟周期性起伏，例如滤波器、走线、连接器造成的通带波纹。

数学上是多个正弦波叠加：

```text
h_ripple_db(f) = sum_i A_i * sin(2*pi*c_i*x + phi_i)
```

其中：

```text
A_i     第 i 个 ripple 分量的幅度
c_i     第 i 个分量在通带内起伏几圈
phi_i   随机初始相位
```

### 4.3 Notch / Bump

`H1_notch_bump_random_generator.py` 模拟局部凹陷或凸起。

直观理解：

```text
某一小段频率突然低一点，叫 notch
某一小段频率突然高一点，叫 bump
```

常见建模方式是 Gaussian：

```text
h_feature_db(f) = A * exp(-0.5 * ((x - c) / sigma)^2)
```

其中：

```text
A       正数表示 bump，负数表示 notch
c       中心位置
sigma   宽度
```

### 4.4 Edge Rolloff

`H1_edge_rolloff_random_generator.py` 模拟通带边缘衰减。

直观理解：

```text
频带中间比较平，靠近边缘时增益下降
```

这对应很多真实硬件中的边缘带宽不足现象。

### 4.5 Measurement Noise

`H1_measurement_noise_random_generator.py` 模拟测量噪声或建模误差。

数学上可以理解为：

```text
h_noise_db(f_k) ~ Normal(0, sigma_db)
```

它不是硬件系统性响应，而是让仿真不要过于理想化。

---

## 5. H1 phase random generation

Phase 不是直接描述幅度，而是描述不同频率分量经过系统后相位转了多少。

复数响应里：

```text
H(f) = |H(f)| * exp(j * phi(f))
```

`phi(f)` 就是 phase。

### 5.1 Linear Phase Delay

`H1_linear_phase_delay_random_generator.py` 模拟一个固定时间延迟。

如果一个系统只是把所有频率都延迟 `tau` 秒，那么频率响应相位是：

```text
phi(f) = -2*pi*f*tau
```

为什么是负号？

因为时域延迟满足：

```text
x(t - tau)  <=>  exp(-j*2*pi*f*tau)
```

所以频率越高，相位下降越快。

### 5.2 Group Delay

Group delay 是相位对角频率的负导数：

```text
tau_g(f) = - d phi(f) / d omega
omega = 2*pi*f
```

单位换算：

```text
tau_g_ns = tau_g_s * 1e9
```

如果：

```text
phi(f) = -2*pi*f*tau
```

那么：

```text
omega = 2*pi*f
phi = -omega*tau
tau_g = -d(-omega*tau)/domega = tau
```

所以一个真实的纯延迟会产生正的 group delay。

当前程序已经把 H1 phase generation 改成更符合物理世界：默认不允许负 delay，并通过 `_enforce_positive_group_delay_phase` 让最终 phase 对应的 group delay 保持物理合理。

### 5.3 Phase Ripple

`H1_phase_ripple_random_generator.py` 模拟相位的周期性波动：

```text
phi_ripple(f) = sum_i A_i * sin(2*pi*c_i*x + phi_i)
```

它会让 group delay 出现波动，因为 group delay 是 phase 的导数。

### 5.4 Local Phase Distortion

`H1_local_phase_distortion_random_generator.py` 模拟局部相位异常，类似某段频率附近出现局部相位扭曲。

可以类比 magnitude 里的 notch/bump，只不过它作用在 phase 上。

### 5.5 Group Delay Ripple

`H1_group_delay_ripple_random_generator.py` 直接从 group delay 的角度生成波动，再积分回 phase。

它的逻辑是：

```text
先随机生成 tau_g(f)
再根据 tau_g(f) = -d phi / d omega
反推出 phi(f)
```

离散形式近似为：

```text
phi[k] = phi[k-1] - tau_g[k] * (omega[k] - omega[k-1])
```

### 5.6 Phase Noise

`H1_phase_noise_random_generator.py` 给 phase 加小随机扰动：

```text
phi_noise[k] ~ Normal(0, sigma_rad)
```

---

## 6. L1-08：幅频 FIR 补偿

L1-08 的目标是补偿 H1 的 magnitude ripple。

### 6.1 H2 Target

对应程序：

```text
L1-08_sim/H2_target_generator.py
```

如果硬件幅度响应是：

```text
|H1(f)|
```

那么理想补偿器应该是它的倒数：

```text
|H2_target(f)| = C / |H1(f)|
```

其中 `C` 是一个归一化常数，用来避免整体增益无意义地变大或变小。

用 dB 表示时，乘法会变成加法：

```text
Htotal_db(f) = H1_db(f) + H2_db(f)
```

为了让总响应平坦，希望：

```text
H1_db(f) + H2_target_db(f) = constant
```

所以：

```text
H2_target_db(f) = constant - H1_db(f)
```

如果取 `constant = 0 dB`，那就是：

```text
H2_target_db(f) = -H1_db(f)
```

### 6.2 FIR Filter

FIR 是 finite impulse response filter，有有限个 tap：

```text
y[n] = sum_{k=0}^{N-1} h[k] * x[n-k]
```

其中：

```text
N       tap 数
h[k]    FIR 系数
x[n]    输入
y[n]    输出
```

频率响应为：

```text
H2(e^{j omega}) = sum_{k=0}^{N-1} h[k] * exp(-j*omega*k)
```

当前 L1-08 使用 real linear-phase FIR。它要求系数左右对称：

```text
h[k] = h[N-1-k]
```

这样做的好处是 FIR 的相位是线性的，不会引入额外的相位畸变。对称 FIR 的 group delay 是常数：

```text
D = (N - 1) / 2
```

### 6.3 Least-Squares FIR Design

对应程序：

```text
L1-08_sim/H2_fir_designer.py
```

因为 tap 数有限，实际 FIR 不可能完美等于 `H2_target`，所以程序用 least squares 找一个最接近目标的系数。

当前代码利用对称 FIR，把未知量减少一半。设：

```text
half_taps = N / 2
D = (N - 1) / 2
```

对称 FIR 的零相位幅度部分可以写成：

```text
A(omega) = sum_{n=0}^{N/2-1} 2*h[n]*cos(omega*(n-D))
```

程序构造矩阵：

```text
B[k,n] = 2*cos(omega_k*(n-D))
```

目标向量是：

```text
t[k] = H2_target_linear(f_k)
```

然后求解：

```text
min_h ||B h - t||^2
```

如果开启 regularization，则求解变成：

```text
min_h ||B h - t||^2 + lambda * ||h||^2
```

这就是 ridge regularization。它的作用是避免系数过大，降低 fixed-point saturation 风险和过拟合风险。

### 6.4 L1-08 评价指标

L1-08 主要看补偿后的 ripple：

```text
ripple_db = max(Htotal_db) - min(Htotal_db)
```

其中：

```text
Htotal_float_db = H1_db + H2_float_db
Htotal_fixed_db = H1_db + H2_fixed_db
```

如果 `ripple_db <= 0.1 dB`，就认为达到当前幅频补偿目标。

---

## 7. L1-08 fixed-point coefficient quantization

对应程序：

```text
L1-08_sim/H2_fixed_point_quantizer.py
```

实际硬件不能无限精度存储 FIR 系数，所以需要 fixed-point 量化。

当前 fixed-point 配置在：

```text
L1_08_experiment_config.json
```

active 配置为：

```text
coeff_total_bits = 16
coeff_frac_bits  = 13
```

这表示系数用 16-bit signed fixed-point，其中 13 bit 是小数位。

量化比例：

```text
scale = 2^frac_bits
```

浮点系数 `c` 先变成整数：

```text
c_int_raw = round(c * scale)
```

signed fixed-point 可表示的整数范围是：

```text
int_min = -2^(total_bits - 1)
int_max =  2^(total_bits - 1) - 1
```

如果超出范围，就 clip：

```text
c_int = clip(c_int_raw, int_min, int_max)
```

再转回浮点用于仿真：

```text
c_fixed = c_int / scale
```

`saturation_count` 表示有多少个系数发生了 clip。

当前 L1-08 quantizer 还保持 FIR 对称性：它会成对量化左右对称的系数，确保：

```text
h_fixed[k] = h_fixed[N-1-k]
```

这样 fixed-point 后仍保持 linear phase FIR 结构。

---

## 8. L1-08 behavior simulation

对应程序：

```text
L1-08_sim/L1_08_behavior_sim.py
```

这一阶段不是只看频率响应曲线，而是生成真实一点的 complex I/Q 多音信号，走一遍链路。

### 8.1 Complex I/Q

I/Q 信号可以写成复数：

```text
x[n] = I[n] + jQ[n]
```

用复数表示后，频域处理非常方便。FFT 后，每个频率 bin 上有一个复数值，表示该频率的幅度和相位。

### 8.2 Multi-tone 输入

程序选择一组 tone bin，在这些频点放置复数 tone：

```text
X[k_i] = A_i * exp(j*theta_i)
```

再用 IFFT 得到时域信号：

```text
x[n] = IFFT(X[k])
```

这模拟多个频率分量同时进入系统。

### 8.3 通过 H1 和 FIR

在频域里，通过 H1 很简单：

```text
Y_H1[k] = X[k] * H1(f_k)
```

再回到时域：

```text
y_H1[n] = IFFT(Y_H1[k])
```

通过 FIR 则在时域卷积：

```text
y_FIR[n] = sum h[m] * y_H1[n-m]
```

当前程序使用 cyclic prefix / block 方式，避免普通卷积边缘瞬态影响测量。

### 8.4 Behavior ripple

程序测量每个 tone 经过系统后的幅度，然后计算：

```text
behavior_ripple_db = max(tone_amp_db) - min(tone_amp_db)
```

它比 dense frequency response 更接近“实际信号进入系统后看到的补偿效果”。

---

## 9. L1-08 QAM EVM simulation

对应程序：

```text
L1-08_sim/L1_08_qam_evm_sim.py
```

QAM 是通信里常见的调制方式。64-QAM 符号可以理解为 I/Q 平面上的点：

```text
s = I + jQ
```

理想情况下，经过系统后再补偿回来，点应该还落在原来的位置。如果偏离，就产生 EVM。

### 9.1 EVM 定义

设参考符号是：

```text
s_ref[k]
```

观测符号是：

```text
s_obs[k]
```

系统可能整体多了一个延迟和复数增益，所以程序先拟合并去掉：

```text
s_equalized[k] = delay_correct_and_gain_normalize(s_obs[k])
```

然后误差为：

```text
e[k] = s_equalized[k] - s_ref[k]
```

EVM 是归一化 RMS error：

```text
EVM_percent = 100 * sqrt(mean(|e[k]|^2) / mean(|s_ref[k]|^2))
```

### 9.2 Magnitude-only EVM

L1-08 只负责幅度补偿，不负责相位补偿。所以当前报告里也看 magnitude-only EVM。

它只比较幅度：

```text
mag_error[k] = |s_obs[k]| / gain_mag - |s_ref[k]|
```

```text
magnitude_only_EVM_percent =
    100 * sqrt(mean(mag_error[k]^2) / mean(|s_ref[k]|^2))
```

因此：

```text
full EVM 会受到 phase distortion 影响
magnitude-only EVM 更适合观察 L1-08 幅频补偿
```

---

## 10. L1-09：group delay 分析

L1-09 的输入不是 L1-08 前的原始时域信号，而是 L1-08 生成的同一个 H1 phase。

对应程序：

```text
L1_09_sim/L1_09_group_delay_analyzer.py
```

它读取：

```text
data/<run>/h1_full_combined_random/together.csv
```

里面有：

```text
freq_hz
h_db
phase_rad
```

### 10.1 为什么要 unwrap phase

计算机里的 `angle()` 通常把相位限制在：

```text
[-pi, pi]
```

但真实相位可能连续下降很多圈，比如：

```text
0, -1, -2, -3, -4, ...
```

如果强行限制在 `[-pi, pi]`，曲线会跳变：

```text
... -3.1, +3.1, ...
```

这种跳变不是真实物理现象，会导致导数计算错误。

`unwrap` 做的事情是：当相邻相位跳变超过 pi 时，自动加减 `2*pi`，把相位恢复成连续曲线。

### 10.2 Group Delay 计算

程序用：

```text
omega = 2*pi*f
phase_unwrapped = unwrap(phase_rad)
group_delay_s = - gradient(phase_unwrapped, omega)
group_delay_ns = group_delay_s * 1e9
```

也就是：

```text
tau_g(f) = - d phi(f) / d omega
```

输出：

```text
group_delay_analysis.csv
group_delay_metrics.csv
phase_before_l1_09.png
group_delay_before_l1_09.png
```

---

## 11. L1-09 all-pass IIR 设计

对应程序：

```text
L1_09_sim/L1_09_allpass_designer.py
```

L1-09 的目标不是改变幅度，而是改变相位 / group delay。所以它使用 all-pass filter。

### 11.1 什么是 all-pass filter

All-pass filter 的特点是：

```text
|A(e^{j omega})| = 1
```

也就是所有频率的幅度都是 1，不改变 magnitude。

但是它的相位不是 0：

```text
angle(A(e^{j omega})) != 0
```

因此它可以改变 group delay。

这正适合 L1-09：

```text
L1-08 已经负责 magnitude
L1-09 只想修 phase / group delay
```

### 11.2 二阶 all-pass section

当前程序使用多级二阶 all-pass。一个二阶 section 可以写成：

```text
A_i(z) = (a2 + a1*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
```

它的极点用半径 `r` 和角度 `theta` 表示：

```text
p = r * exp(j*theta)
p* = r * exp(-j*theta)
```

为了稳定，必须：

```text
0 < r < 1
```

对应系数关系可以写成：

```text
a1 = -2*r*cos(theta)
a2 = r^2
```

多级 all-pass 是多个 section 相乘：

```text
A_total(z) = product_i A_i(z)
```

它仍然是 all-pass：

```text
|A_total(e^{j omega})| = 1
```

### 11.3 为什么用多个二阶 section

一个二阶 section 只能在某个频率附近产生一段 group delay 形状。真实硬件的 group delay 可能有多个峰谷，所以需要多个 section 叠加。

当前配置为：

```text
sections = 8
```

也就是 8 个二阶 all-pass section。

### 11.4 L1-09 优化目标

L1-09 希望：

```text
H1_group_delay(f) + allpass_group_delay(f) ≈ target_delay
```

其中 `target_delay` 是一个常数，程序会自动优化。它通常取比原始最大 group delay 更晚一点的值，因为 all-pass filter 只能增加延迟形状，不能让信号真正提前。

程序先对原始 group delay 做平滑：

```text
fit_delay_ns = moving_average(input_group_delay_ns)
```

再设置目标下界：

```text
target_delay_ns >= max(fit_delay_ns) + margin
```

优化变量包括：

```text
r_i
theta_i
candidate_target_delay_ns
```

目标函数是：

```text
residual(f_k) =
    (fit_delay_ns(f_k) + allpass_delay_ns(f_k) - candidate_target_delay_ns)
    / scale
```

优化问题是：

```text
min sum_k residual(f_k)^2
```

程序使用：

```text
scipy.optimize.least_squares
```

输出：

```text
allpass_coefficients.csv
allpass_response.csv
allpass_metrics.csv
group_delay_before_after_l1_09.png
phase_before_after_l1_09.png
```

---

## 12. L1-09 fixed-point quantization

对应程序：

```text
L1_09_sim/L1_09_fixed_point_quantizer.py
```

L1-09 all-pass 也是硬件算法，所以系数不能只停留在浮点。当前配置在：

```text
L1_09_experiment_config.json
```

active 配置：

```text
coeff_total_bits = 18
coeff_frac_bits  = 15
```

量化公式和 L1-08 类似：

```text
scale = 2^frac_bits
c_int_raw = round(c * scale)
c_int = clip(c_int_raw, int_min, int_max)
c_fixed = c_int / scale
```

但 L1-09 有一个非常重要的额外要求：保持 all-pass 结构。

程序只量化 denominator feedback terms，然后把它们镜像到 numerator：

```text
denominator: 1 + a1*z^-1 + a2*z^-2
numerator:   a2 + a1*z^-1 + 1*z^-2
```

这样 fixed-point 后仍然保持 all-pass 结构，幅度不会因为系数量化而明显偏离 1。

### 12.1 Stable=True 衡量什么

L1-09 是 IIR，有反馈。IIR 稳定性由 pole 决定。

稳定条件：

```text
max(|pole|) < 1
```

所以：

```text
stable=True
```

表示 fixed-point 后所有 pole 仍在单位圆内。

它只说明滤波器不会发散，不代表 group delay 已经补偿得很好，也不代表 EVM 一定好。

### 12.2 为什么 pole 和 denominator 的根有关

这一节解释一个核心问题：为什么 IIR filter 的稳定性要看 denominator 的根，而不是随便看某一个输出样本。

先从最简单的一阶反馈系统开始：

```text
y[n] = a * y[n-1]
```

这里先假设没有输入，只看系统自己的自然响应。这个假设不是说真实系统没有输入，而是为了单独观察反馈结构本身会不会把过去的输出放大。

如果初始输出是：

```text
y[0] = C
```

那么后面的输出是：

```text
y[1] = a * C
y[2] = a^2 * C
y[3] = a^3 * C
...
y[n] = a^n * C
```

所以稳定性不取决于例子里选的是 `y[0] = 1`，而是取决于：

```text
a^n 是否会随着 n 增大而趋近于 0
```

如果：

```text
|a| < 1
```

那么 `a^n -> 0`，过去输出的影响会越来越小，系统稳定。

如果：

```text
|a| = 1
```

那么过去输出不会衰减，系统可能持续振荡，属于边界情况。

如果：

```text
|a| > 1
```

那么 `a^n` 会越来越大，系统会发散。

这就是 pole 稳定性判断的直观来源。

对于 L1-09 这种二阶 IIR all-pass section，每一级可以写成：

```text
H_i(z) = (a2 + a1 z^-1 + z^-2) / (1 + a1 z^-1 + a2 z^-2)
```

其中 denominator 是：

```text
A(z) = 1 + a1 z^-1 + a2 z^-2
```

pole 就是让 denominator 等于 0 的解：

```text
A(z) = 0
```

也就是：

```text
1 + a1 z^-1 + a2 z^-2 = 0
```

为了更容易看，把两边乘以 `z^2`：

```text
z^2 + a1 z + a2 = 0
```

这个二次方程有两个根：

```text
p1, p2
```

这两个根就是这个二阶 section 的两个 pole。

为什么它们决定稳定性？因为 denominator 来自反馈项。二阶 IIR 的时域形式是：

```text
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
       - a1*y[n-1] - a2*y[n-2]
```

其中：

```text
-a1*y[n-1] - a2*y[n-2]
```

就是反馈。为了看系统自己的自然响应，先令输入为 0：

```text
x[n] = 0
```

于是：

```text
y[n] = -a1*y[n-1] - a2*y[n-2]
```

假设自然响应的形式是：

```text
y[n] = C * r^n
```

把它代入递推关系：

```text
C*r^n = -a1*C*r^(n-1) - a2*C*r^(n-2)
```

两边除以 `C*r^(n-2)`：

```text
r^2 = -a1*r - a2
```

整理后得到：

```text
r^2 + a1*r + a2 = 0
```

这个方程和刚才 denominator 乘以 `z^2` 后得到的方程完全一样：

```text
z^2 + a1*z + a2 = 0
```

所以：

```text
denominator 的根 = pole = 自然响应里的增长/衰减因子 r
```

如果两个根都满足：

```text
|p1| < 1
|p2| < 1
```

那么：

```text
p1^n -> 0
p2^n -> 0
```

系统自然响应会衰减，所以稳定。

如果任意一个根满足：

```text
|p| >= 1
```

那么对应的自然响应不会衰减，甚至会放大，IIR filter 就可能持续振荡或发散。

因此，程序里的：

```text
stable=True
```

准确含义是：

```text
量化后的 L1-09 all-pass IIR filter 的所有 pole 都在单位圆内。
```

如果 L1-09 使用 8 个二阶 all-pass section，那么每一级有 2 个 pole，总共有：

```text
8 * 2 = 16 个 pole
```

程序需要确认这 16 个 pole 全部满足：

```text
|pole| < 1
```

只有这样，fixed-point 系数量化后的 IIR all-pass filter 才能被认为是数值稳定的。

---

## 13. L1-09 EVM_LIN 分析

对应程序：

```text
L1_09_sim/L1_09_evm_lin_calculator.py
```

这个模块不生成完整时域 QAM，而是直接基于频率响应估算线性系统误差。

它比较三个阶段：

```text
after_h1
after_l1_08_fixed_fir
after_l1_08_fixed_fir_plus_l1_09_allpass
```

频率响应分别是：

```text
R1(f) = H1(f)
R2(f) = H1(f) * H2_fixed(f)
R3(f) = H1(f) * H2_fixed(f) * A_l1_09(f)
```

其中 `A_l1_09(f)` 是 all-pass response。

### 13.1 去掉整体 gain 和 delay

系统整体增益和整体延迟通常不是我们关心的畸变。程序会拟合并去掉一个复数 gain 和一个线性 delay：

```text
R_equalized(f) = R(f) corrected by fitted gain and fitted delay
```

然后计算它和理想值 1 的偏差：

```text
residual(f) = R_equalized(f) - 1
```

### 13.2 EVM_LIN

```text
EVM_LIN_percent = 100 * sqrt(mean(|residual(f)|^2))
```

它还拆成：

```text
magnitude_only_evm_percent
phase_only_evm_percent
```

这样可以观察 L1-08 主要改善 magnitude，L1-09 主要改善 phase。

---

## 14. L1-09 QAM EVM validation

对应程序：

```text
L1_09_sim/L1_09_qam_evm_validator.py
```

这个模块比 EVM_LIN 更接近实际信号。它生成 QAM-loaded IF 信号，然后依次经过：

```text
H1
L1-08 fixed FIR
L1-09 all-pass IIR
```

对比三种状态：

```text
after_h1
after_l1_08_fixed
after_l1_08_plus_l1_09
```

其中 L1-09 all-pass 是真正以 IIR SOS 的形式作用在时域 I/Q 信号上。

### 14.1 IIR SOS 级联

SOS 是 second-order sections。每个二阶 section 对应：

```text
y[n] + a1*y[n-1] + a2*y[n-2]
    = b0*x[n] + b1*x[n-1] + b2*x[n-2]
```

多级 SOS 就是前一级输出作为后一级输入。

### 14.2 QAM EVM

QAM EVM 仍然使用：

```text
EVM_percent = 100 * sqrt(mean(|s_equalized - s_ref|^2) / mean(|s_ref|^2))
```

程序同时输出：

```text
l1_09_qam_evm_summary.csv
l1_09_qam_per_bin.csv
l1_09_qam_evm.png
```

---

## 15. Sweep test 是什么

当前完整 pipeline 不是 sweep。完整 pipeline 只跑一组 active config。

Sweep test 是系统性改变参数，批量观察算法是否稳定。

相关目录：

```text
sweep_test/
sweep_test_config.json
```

典型 sweep 维度包括：

```text
bandwidth profile
seed case
tap_num
regularization
fixed-point format
```

每个 combo 会产生自己的：

```text
data/
graph/
logs/
```

再被复制到：

```text
sweep_result/
```

当前用户明确区分：

```text
run_full_l1_08_l1_09_pipeline.py  是完整 pipeline
sweep_test/run_sweep.py           是 sweep test
```

---

## 16. 当前配置的含义

### 16.1 Active profile

当前 active L1-08 配置大致是：

```text
frequency range: 3.5 GHz - 4.5 GHz
num_points:      1001
fs_hz:           12 GHz
FIR tap_num:     64
regularization:  1e-4
FIR fixed-point: 16 total bits, 13 frac bits
```

当前 L1-09 配置大致是：

```text
all-pass sections: 8
smooth_window:     31
fixed-point:       18 total bits, 15 frac bits
```

### 16.2 Seed

Seed 不是物理大小，不是“越大越复杂”。Seed 只是随机数生成器的起点。

如果：

```text
seed 相同
config 相同
```

那么随机生成的 H1 就相同。

如果：

```text
seed 不同
```

那么会生成另一条随机 H1，用来验证算法是否只对某一条随机曲线有效。

---

## 17. 程序输出应该怎么看

一次完整 pipeline 结束后，重点看：

```text
data/<run>/run_summary.json
```

它汇总了每个 stage 的指标。

L1-08 重点看：

```text
l1_08_h2_fir_design / ripple_after_db
l1_08_h2_fixed_point / ripple_after_fixed_db
l1_08_behavior / ripple_after_fir_fixed_db
l1_08_qam_evm / after_fixed_fir_magnitude_only_evm_percent
```

L1-09 重点看：

```text
l1_09_fix_group_delay / group_delay_ripple_pp_ns
l1_09_fix_allpass_iir_fs / compensated_group_delay_ripple_pp_ns
l1_09_fix_allpass_iir_fixed / stable, saturation_count, max_pole_radius
l1_09_fix_evm_lin_fixed / phase_only_evm_percent
l1_09_fix_qam_evm_iir_fixed / after_l1_08_plus_l1_09_evm_percent
```

图像都在：

```text
graph/<run>/
```

---

## 18. 当前 pipeline 的工程边界

当前程序是 behavior-level simulation，不是完整 RTL simulation。

已经包含：

```text
1. 复数 H1 magnitude + phase 随机建模
2. L1-08 FIR 设计
3. L1-08 FIR coefficient fixed-point quantization
4. L1-08 multi-tone 行为仿真
5. L1-08 QAM EVM 辅助验证
6. L1-09 group delay 分析
7. L1-09 floating all-pass 设计
8. L1-09 all-pass coefficient fixed-point quantization
9. L1-09 EVM_LIN 和 QAM EVM 验证
```

尚未完整包含：

```text
1. RTL cycle-accurate fixed-point 仿真
2. FIR accumulator bit-growth / rounding / truncation 细节
3. IIR 内部状态 fixed-point 量化
4. IIR runtime overflow / saturation
5. 真实硬件测量数据直接导入
6. 与仪器实测数据逐点对比
```

所以当前阶段的结论应表述为：

```text
算法方向和行为级模型正在验证中。
当前 fixed-point 主要是 coefficient quantization。
它还不能完全等价于最终 RTL fixed-point implementation。
```

### 18.1 后续 RTL fixed-point 仿真需要补充什么

当前 L1-08 + L1-09 pipeline 暂时不把完整 RTL fixed-point 作为主要目标。原因是当前阶段的重点是先确认算法行为级链路是否合理，也就是：

```text
H1 random channel
        ↓
L1-08 magnitude FIR compensation
        ↓
L1-09 all-pass group-delay compensation
        ↓
behavior-level metric checking
```

当前 fixed-point 已经覆盖的是 coefficient quantization，也就是把 FIR / all-pass 的浮点系数量化成有限 bit 数值后，再检查补偿效果、saturation 和稳定性。这一步可以回答：

```text
如果 filter coefficient 不能无限精度保存，
而只能使用 fixed-point coefficient，
当前算法是否仍然大致有效？
```

但是完整 RTL fixed-point 不只是“系数量化”。真正硬件实现时，每一级运算都会受到有限位宽影响。后续进入 RTL 阶段时，需要重新回到 simulation，补充更接近硬件的 bit-true 或 RTL-like fixed-point scenario：

```text
input I/Q quantization
        ↓
每一级 all-pass filter 的乘法 fixed-point
        ↓
加法器 / accumulator fixed-point
        ↓
内部 delay register fixed-point
        ↓
每一级输出 rounding / truncation / saturation
        ↓
多级 SOS cascade 后的最终输出
```

这部分后续需要重点检查：

```text
1. 输入 I/Q 被量化后，phase extraction 和 group delay 是否仍可靠
2. 每一级 second-order all-pass section 内部乘法是否会溢出
3. accumulator 位宽是否足够，是否需要 guard bits
4. feedback path 中的 rounding / truncation 是否引入 limit cycle 或额外噪声
5. 多级 SOS cascade 的 section ordering 和 scaling 是否影响稳定性
6. 最终 fixed-point 输出对应的 phase residual、group-delay ripple 和 EVM 是否仍满足指标
```

因此，当前 report 中的 fixed-point 结论应理解为：

```text
当前完成的是 coefficient-quantized behavior-level fixed-point analysis。
完整 RTL-level fixed-point arithmetic simulation 暂不在当前阶段展开，
后续开始 RTL 实现时再回到 simulation 中彻底补齐。
```

---

## 19. 一句话总结

当前程序完整模拟了一个“硬件链路 H1 产生幅度和相位畸变，L1-08 用 FIR 修幅度，L1-09 用 all-pass IIR 修 group delay”的行为级算法链路。

L1-08 的数学核心是：

```text
H2_target_db = constant - H1_db
min_h ||B h - H2_target||^2 + lambda||h||^2
```

L1-09 的数学核心是：

```text
tau_g(f) = -d phi / d omega
H_allpass magnitude = 1
min_params ||tau_H1(f) + tau_allpass(f) - target_delay||^2
```

最终用 dense response、multi-tone I/Q、QAM EVM、fixed-point saturation 和 IIR stability 等指标来判断当前算法是否合理。
