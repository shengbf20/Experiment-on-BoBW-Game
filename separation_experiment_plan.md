# Separation Example 实验方案与进度

## 1. 目标与口径

实验考察论文 Appendix A 的非平稳 separation example：同一条合法的冻结
open-loop opponent sequence 上，

\[
V_T(u^\star)=O(1),\qquad E_T=\Theta(T),\qquad G_T=O(1).
\]

该结论分离的是 comparator-local、realized-gradient-variation 和 worst-case
certificate，不是算法性能排名，也不是 realized regret superiority。

精确博弈、opponent 递归和渐近证明以 `../note/v1/appendix_a.tex` 为准；实现位于
`src/separation.py`。旧 stationary G3 不是本实验。

---

## 2. Part A：冻结序列与机制验证

### 2.1 协议

对每个偶数 horizon

\[
T\in\{200,500,1000,2000,5000,10000,20000\},
\]

执行：

1. 从规定初始化运行确定性的 D005，递归生成完整 opponent sequence
   \(y_{1:T}\)；
2. 冻结该 sequence；
3. 从头初始化 D005，只回放冻结的 \(y_{1:T}\)；
4. 检查生成与回放的 \(x_t,g_t\) 在绝对误差 \(10^{-12}\) 内一致；
5. 保存 \(x_t,g_t,y_t,V_t,E_t,G_t,R_t\)。

定义

\[
V_T(u^\star)=\sum_{t=2}^T
\|\nabla_x\Phi(u^\star,y_t)-\nabla_x\Phi(u^\star,y_{t-1})\|^2,
\]

\[
E_T=\|g_1\|^2+\sum_{t=2}^T\|g_t-g_{t-1}\|^2,
\qquad
G_T=\max_t\|g_t\|.
\]

### 2.2 分层成功判据

Part A 不再用一个布尔值混合“实现正确”和“原始渐近斜率可见”。结果必须分别报告：

- `implementation_valid`：冻结回放一致、\(x_1=0\)、opponent 坐标合法；
- `V_T_is_one`：所有 horizon 上 \(V_T(u^\star)=1\)；
- `G_T_O1`：\(G_T\) 位于理论常数界内；
- `linear_tail_mechanism_valid`：settled tail 的每轮平方梯度跳变接近证明中的正常数，
  且 tail contribution 随 tail 长度线性增长；
- `raw_ET_scaling_observed`：仅当原始 \(E_T\) 对 \(T\) 的有限-horizon 图确实显示
  线性 scaling 时才可为真。

`part_A_closed` 只表示协议与线性机制已经验证，不表示
`raw_ET_scaling_observed=true`。

### 2.3 已完成结果

Part A 已完成，且未改写已有 G1/G2、warm-restart 或 \(L_F\) 结果：

- 生成与冻结回放一致；
- 每个 horizon 都有 \(V_T(u^\star)=1\)；
- \(T\ge500\) 时 \(G_T\approx2.076\)，保持常数量级；
- settled tail 的理论平方跳变常数为
  \(c\approx6.592\times10^{-7}\)；
- \(T=5000,10000,20000\) 的最后四分之一 variation contribution 分别约为
  \(8.03\times10^{-4},1.66\times10^{-3},3.32\times10^{-3}\)，与 tail 长度
  近似成正比。

但在计划 horizon 上，原始 \(E_T\approx5.6\) 仍由 \(O(1)\) 瞬态主导，
`raw_ET_scaling_observed=false`。近似分解为

\[
E_T\approx C+cT,\qquad C\approx5.62,
\]

线性项与瞬态相当约需 \(T\approx8.5\times10^6\)。因此不通过放大 \(\eta\) 改写
理论实例，也不为展示渐近斜率而默认追加超长跑。

### 2.4 论文可用叙述

可以声称：实验验证了合法冻结回放、精确 \(V_T(u^\star)=1\)、有界 \(G_T\)，以及
证明中产生 \(E_T=\Theta(T)\) 的正 tail-density 机制。

不得声称：计划 horizon 的原始 \(E_T\) 曲线直接观察到 \(\Theta(T)\)，或实际
\(\sqrt{E_T}\) 已呈现 \(\Theta(\sqrt T)\)。渐近结论来自 Appendix A 的证明。

实际 regret \(R_T(u^\star)\) 只保留在结果中；负 regret 合法，但不用于证明其绝对值
为 \(O(1)\)，也不用于算法排名。

### 2.5 Part A 产物

- `results/exp_sep_T*.json/.npz`：每个 horizon 的摘要、冻结 \(y\) 和完整轨迹；
- `results/exp_sep_partA.json`：分层 verdict；
- `figures/exp_sep_partA.pdf/.png`：唯一正式候选图，展示 \(V_T,G_T\)、线性 tail
  contribution 和最长 horizon 的两段 settled trajectory。

旧的 variation / certificate / mechanism 三图不再保留。尤其不能把
\(\sqrt{T\cdot\text{late }\Delta g^2}\) 称作实际 certificate。

---

## 3. Part B：D005 vs. Hsieh（可选）

### 3.1 开始条件

Part A 已满足开始 Part B 所需条件：每个 NPZ 都保存了冻结的 `y`，无需重新生成
opponent，也不得依据 Hsieh 的轨迹修改该 sequence。汇总中的每一行同时记录规范化
`frozen_y_sha256`；Part B 读取 `y` 后必须复算并核对该 hash。

Hsieh 实现已在此前实验精简中删除。开展 Part B 时应新增隔离的 baseline 模块和检查，
不要把 Hsieh 状态混入 `ClosedFormPlayer` 或 Part A 生成逻辑。

### 3.2 公平比较

对每个已保存 horizon：

1. 从 `exp_sep_T*.npz` 读取同一个冻结 \(y_{1:T}\)；
2. 分别从规定初值运行 D005 与 Hsieh / Euclidean OptDA；
3. 比较同一 comparator \(u^\star=1\) 下的
   \(R_T^{\mathrm{D005}}(u^\star)\)、
   \(R_T^{\mathrm{Hsieh}}(u^\star)\) 及可选的 \(R_T/T\)；
4. 单独记录 baseline 的实现 sanity checks 与数值稳定性。

### 3.3 纳入论文的判据

- D005 明显更小且差距随 \(T\) 稳定扩大：可作为加分实验，但只能描述这组按 D005
  离线构造的 sequence；
- 二者接近或 Hsieh 更好：不影响 Part A，只记录在实验日志；
- 若结果说明 “certificate separation 不等于 realized-regret separation”，可考虑作为
  appendix limitation；
- 无论结果如何，都不得表述为一般性的算法 superiority。

---

## 4. 命令

运行 Part A learner：

```text
python scripts/exp_separation.py
```

只用已保存结果重建 verdict 与图：

```text
python scripts/exp_separation.py --assemble
python scripts/plot_separation.py
```

Part B 尚无执行入口；下一阶段从读取现有冻结 `y` 和隔离实现 baseline 开始。
