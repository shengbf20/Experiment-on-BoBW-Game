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

## 2. Part A：改进 construction 后的有限样本 separation

原始数值版本取 \(u^\star=1\)、\(\eta=\delta_\star/16\)，虽然渐近证明成立，
但 \(|g_1|^2\) 等 \(O(1)\) transient 约为 \(5.6\)，而 settled tail 的每轮平方
跳变仅约 \(6.59\times10^{-7}\)。该版本已被下面的严格等价改进替代：

\[
u^\star=\frac14,\qquad \eta=\delta_\star.
\]

平移使 prescribed initialization \(x_1=0\) 更接近 comparator，且
\(u^\star\ne0\) 仍保留 \(\|u^\star\|\sqrt{E_T}\) 的 polynomial order。
增大 \(\eta\) 将理论平方跳变常数放大 \(16^2\) 倍。仓库已检查改进后的
convexity/concavity margins、\(\eta=\delta_\star\) 以及 \(13/48\) annulus
fraction 仍为正；这只说明没有发现 construction 修改导致这些证明条件失效，
不是对 Appendix A 全文的完整审计。`../note/v1/appendix_a.tex` 是当前证明文本。

### 2.1 协议

对每个偶数 horizon

\[
T\in\{200,500,1000,2000,5000,10000,20000,50000,100000\},
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
- `raw_ET_scaling_observed`：在预先指定的
  \(10^4\le T\le10^5\) 窗口内，原始 \(E_T\) 的 log-log slope 必须位于
  \([0.75,1.25]\)，且该窗口内 \(\max(E_T/T)/\min(E_T/T)\le2\)；
- `main_text_candidate`：上述实现、机制和原始 scaling 判据必须同时通过。

`part_A_closed` 表示协议与理论机制通过；是否足以进入正文另由
`main_text_candidate` 控制。

### 2.3 已完成结果

改进版 Part A 已完成，且未改写已有 G1/G2、warm-restart 或 \(L_F\) 结果：

- 生成与冻结回放一致；
- 每个 horizon 都有 \(V_T(u^\star)=1\)；
- \(G_T\) 从约 \(1.19\) 稳定到约 \(0.98\)，保持常数量级；
- settled tail 的理论平方跳变常数为
  \(c\approx1.6876\times10^{-4}\)；
- 原始 \(E_T\) 在 \(T=10^4,2\times10^4,5\times10^4,10^5\) 时分别约为
  \(2.685,4.379,9.462,17.933\)；
- 目标窗口内原始 \(E_T\) 的 log-log slope 为 \(0.827\)，
  \(E_T/T\) 从 \(2.685\times10^{-4}\) 稳定到 \(1.793\times10^{-4}\)；
- settled tail contribution 的 log-log slope 为 \(1.016\)。

数值分解约为

\[
E_T\approx C+cT,\qquad C\approx1.06,
\]

对应 crossover 约为 \(T\approx6.27\times10^3\)。所有预设判据均通过：
`raw_ET_scaling_observed=true`、`main_text_candidate=true`。

### 2.4 论文可用叙述

正文 5.3 应题为 **Finite-Horizon Illustration of the Separation Example**，
只报告：

\[
V_T(u^\star)=1,\qquad G_T=O(1),
\]

以及 \(E_T\) exhibits near-linear finite-horizon growth over
\(10^4\le T\le 10^5\)，with \(E_T/T\) approaching a positive constant。
\(\sqrt{E_T}\) 的增长可写成与渐近 \(\Theta(\sqrt T)\) prediction consistent，
但实验并不 establishes 该 rate。紧接着写：

> The rigorous asymptotic statement \(E_T=\Theta(T)\), and hence the
> \(\Theta(\sqrt T)\) scaling of the corresponding last-gradient term,
> follows from Section 4 / Appendix A rather than from the finite-horizon fit.

不要写“\(E_T\) 贴 \(cT\)”：\(T=10^4\) 时仍有明显 transient，到 \(10^5\) 才相当接近。
图标题使用 finite-horizon separation，不要写成 \(E_T\propto T\)。

实际 regret \(R_T(u^\star)\) 只保留在结果中；负 regret 合法，但不用于证明其绝对值
为 \(O(1)\)，也不用于算法排名。

### 2.5 Part A 产物

- `results/exp_sep_T*.json/.npz`：每个 horizon 的摘要、冻结 \(y\) 和完整轨迹；
- `results/exp_sep_partA.json`：分层 verdict；
- `figures/exp_sep_partA.pdf/.png`：唯一正式候选图，展示原始
  \(V_T,G_T,E_T\)、\(E_T/T\) 趋向正常数，以及最长 horizon 的两段 settled trajectory。
  左图标题为 finite-horizon separation of \(V_T(u^\star)\) and \(E_T\)。

旧版 construction 的冻结序列和三张诊断图不再保留。图中展示的是实际原始
\(E_T\)，没有把 tail proxy 伪装成 certificate。

---

## 3. Part B：D005 vs. Hsieh（可选）

### 3.1 开始条件

Part A 已正式收尾：措辞与 `../note/skeleton.md` §5.3 对齐，每个 NPZ 都保存了冻结的 `y`，无需重新生成
opponent，也不得依据 Hsieh 的轨迹修改该 sequence。汇总中的每一行同时记录规范化
`frozen_y_sha256`；Part B 读取 `y` 后必须复算并核对该 hash。

Hsieh 实现已在此前实验精简中删除。开展 Part B 时应新增隔离的 baseline 模块和检查，
不要把 Hsieh 状态混入 `ClosedFormPlayer` 或 Part A 生成逻辑。

### 3.2 公平比较

对每个已保存 horizon：

1. 从 `exp_sep_T*.npz` 读取同一个冻结 \(y_{1:T}\)；
2. 分别从规定初值运行 D005 与 Hsieh / Euclidean OptDA；
3. 比较同一 comparator \(u^\star=1/4\) 下的
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

实际结果走第二条：Hsieh 更好但差距饱和。第三条的 appendix limitation 选项已否决，
因为对照序列不公平、相对差距只有 \(1.5\%\)，放进附录容易被误读成算法排名。

### 3.4 已完成结果

Part B 已按冻结回放协议完成。Hsieh 实现隔离在 `src/hsieh.py`，不导入
`ClosedFormPlayer`，也不参与 opponent 生成。每个 horizon 都核验了
`frozen_y_sha256`，D005 回放与 Part A 的 \(x_t,g_t\) 在 \(10^{-12}\) 内一致，
Hsieh 保持 \(x_1=0\)、有限状态、非降 \(\lambda_t\)，且
\(\lambda_T=\sqrt{\tau+E_T^{\mathrm{Hsieh}}}\)。G2 self-play smoke 恢复了历史
终端 \(\mathrm{Reg}^x(a)\approx0.377\)、\(Q_T\approx11.48\)。

同一 comparator \(u^\star=1/4\) 上，Hsieh 的 \(R_T\) 在全部 horizon 都更小
（更负），但加性差距在 \(T\gtrsim2\times10^3\) 后饱和在约 \(-63\)，并不随
\(T\) 扩大。目标窗口内：

| \(T\) | \(R_T^{\mathrm{D005}}\) | \(R_T^{\mathrm{Hsieh}}\) | 差距 | \(R_T^{\mathrm{D005}}/T\) | \(R_T^{\mathrm{Hsieh}}/T\) |
|---:|---:|---:|---:|---:|---:|
| \(10^4\) | \(-342.7\) | \(-406.6\) | \(-63.9\) | \(-3.43\times10^{-2}\) | \(-4.07\times10^{-2}\) |
| \(2\times10^4\) | \(-753.8\) | \(-817.4\) | \(-63.6\) | \(-3.77\times10^{-2}\) | \(-4.09\times10^{-2}\) |
| \(5\times10^4\) | \(-1987.1\) | \(-2050.1\) | \(-63.0\) | \(-3.97\times10^{-2}\) | \(-4.10\times10^{-2}\) |
| \(10^5\) | \(-4042.5\) | \(-4104.7\) | \(-62.2\) | \(-4.04\times10^{-2}\) | \(-4.10\times10^{-2}\) |

\(T=10^5\) 时相对差距仅 \(1.5\%\)。两者都是大幅负 regret，平均 regret 趋向同一
负常数。Hsieh 的 \(E_T\) 更大（约 \(124\) vs \(17.9\)），\(G_T\) 约 \(2.61\)
且有界，\(x_T\) 与 D005 同在 \(0.41\) 附近。

分层判据：`implementation_valid=true`，
`bonus_experiment_candidate=false`，
`hsieh_better_or_comparable=true`，
`certificate_not_realized_separation=true`。

**论文决策（2026-09-13）：不写入实验。** 对照序列由 D005 离线构造，Hsieh 仅有
饱和的 \(O(1)\) 加性优势，相对差距在 \(T=10^5\) 只有 \(1.5\%\)，两边 \(R_T/T\)
趋向同一负常数。这既不是 D005 加分实验，也不适合作为 appendix limitation
图：读者容易误读成算法排名。Part A 不受影响。完整局限性见
`../note/experiment_log.md`。

### 3.5 Part B 产物（仅本地 / 日志，非论文实验）

- `src/hsieh.py`：隔离的 Euclidean OptDA（\(\tau=1\)，不拟合）；
- `scripts/check_hsieh.py`：代数恒等式、模块隔离、G2 self-play smoke；
- `results/exp_sep_partB_T*.json/.npz`、`exp_sep_partB.json`：本地摘要，不进入论文图；
- 若生成 `figures/exp_sep_partB.*`，也只是诊断图，不跟踪为论文图。

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

Part B 读取已保存冻结 `y`，回放 D005 与 Hsieh，不重新生成 opponent：

```text
python scripts/check_hsieh.py
python scripts/exp_sep_partB.py
python scripts/plot_sep_partB.py
```

只用已保存 Part B 结果重建 verdict 与图：

```text
python scripts/exp_sep_partB.py --assemble
python scripts/plot_sep_partB.py
```

Part A 的措辞、冻结输入与正文 5.3 候选地位不变。Part B 已关闭为日志记录：
不得写入正文或附录实验，不得按 baseline 轨迹重新构造 opponent，也不得改写
已有 G1/G2、warm-restart 或 \(L_F\) 结果。
