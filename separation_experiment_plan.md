# Separation Example 实验执行方案

## 1. 实验目的

本实验用于数值展示 separation example 的核心现象：

\[
V_T(u^\star)=O(1),\qquad
E_T=\Theta(T),\qquad
G_T=O(1),
\]

从而说明 comparator-local variation certificate 可以在同一条合法 opponent trajectory 上显著小于基于 realized gradient variation 的 certificate。

实验主要回答：

> 为什么研究 \(V_T(u)\)-dependent fallback 是有意义的？

注意：该实验首先是 **certificate separation**，不是 realized regret superiority 的证明。

---

## 2. 实验步骤

### Part A：Without Bonus —— 核心 separation 实验

这部分必须完成并进入论文。

#### A1. Horizon 设置

取若干偶数 horizon，例如：

\[
T\in\{200,500,1000,2000,5000,10000,20000\}.
\]

#### A2. 生成固定 opponent sequence

对每个 \(T\)：

1. 按 separation example 的构造运行 D005；
2. 离线生成完整 opponent sequence \(y_{1:T}\)；
3. 生成后将该 sequence 冻结；
4. 从头 replay D005，确保正式实验面对的是预先固定的 open-loop sequence。

#### A3. 记录量

每轮记录：

- learner action \(x_t\)；
- gradient \(g_t=
abla_x\Phi(x_t,y_t)\)。

最终计算：

\[
V_T(u^\star)
=
\sum_{t=2}^T
\|
abla_x\Phi(u^\star,y_t)-
abla_x\Phi(u^\star,y_{t-1})\|^2,
\]

\[
E_T
=
\|g_1\|^2+
\sum_{t=2}^T
\|g_t-g_{t-1}\|^2,
\]

\[
G_T=\max_t\|g_t\|,
\]

以及 D005 的实际 regret：

\[
R_T(u^\star)
=
\sum_{t=1}^T
igl[
\Phi(x_t,y_t)-\Phi(u^\star,y_t)
igr].
\]

#### A4. 成功判据

应观察到：

\[
V_T(u^\star)pprox 1,
\]

\[
E_T/T
\]

保持在正的常数量级，即 \(E_T=\Theta(T)\)，同时

\[
G_T=O(1).
\]

如果这三点成立，则核心 separation 现象得到清晰数值展示。

---

### Part B：With Bonus —— D005 vs. Hsieh

这部分是可选加分项；结果有解释价值时再进入论文。

#### B1. 公平比较方式

对每个 \(T\)：

1. 使用 Part A 已经生成并冻结的同一个 opponent sequence \(y_{1:T}\)；
2. 分别运行：
   - D005；
   - Hsieh / OptDA baseline；
3. 两者面对完全相同的 loss sequence
   \[
   f_t(x)=\Phi(x,y_t).
   \]

#### B2. 比较指标

主要比较：

\[
R_T^{\mathrm{D005}}(u^\star),
\qquad
R_T^{\mathrm{Hsieh}}(u^\star).
\]

可同时报告：

\[
R_T/T.
\]

#### B3. 如何决定是否写入论文

- 若 D005 的 realized regret 明显更小，且差距随 \(T\) 稳定扩大：可作为正文加分实验；
- 若二者接近：通常不放正文；
- 若 Hsieh 更好：不影响 Part A 的结论，可不纳入论文；
- 若结果虽无 superiority，但能帮助解释“certificate separation ≠ realized-regret separation”，可考虑放 appendix。

注意：该 opponent sequence 是依据 D005 的确定性轨迹离线构造的，因此即使 D005 优于 Hsieh，也不应表述为一般性的算法 superiority。

---

## 3. 实验产出

### 核心产出 1：Variation Separation Figure

横轴：\(T\)。

建议展示：

\[
V_T(u^\star),
\qquad
E_T.
\]

最好同时加入：

\[
E_T/T
\]

或参考线 \(O(1)\)、\(O(T)\)。

目标：直观看到

\[
V_T(u^\star)=O(1),
\qquad
E_T=\Theta(T).
\]

### 核心产出 2：Certificate Scaling Figure

展示：

\[
\sqrt{V_T(u^\star)},
\qquad
\sqrt{E_T},
\qquad
G_T\sqrt T.
\]

目标：直观看到

\[
O(1)
\quad	ext{vs.}\quad
\Theta(\sqrt T).
\]

### 可选产出 3：Actual Regret Comparison

仅在结果有价值时生成或纳入论文：

\[
R_T^{\mathrm{D005}}(u^\star)
\quad	ext{vs.}\quad
R_T^{\mathrm{Hsieh}}(u^\star).
\]

---

## 4. 最终执行原则

- **Part A 是核心实验，必须成功。**
- **Part B 是 bonus，不决定 separation example 是否成立。**
- 论文主叙述应强调 certificate separation，而不是把实验包装成算法性能排名。
- 建议直接复用现有 `Experiment-on-BoBW-Game` 仓库，在其中新增独立 separation experiment 模块和输出目录。
