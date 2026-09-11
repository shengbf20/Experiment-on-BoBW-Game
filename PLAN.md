# 实验方案（flagship = `thm:t006` / D005）

实验是插图，不是主结果。对准 ICLR / ICML 主文约 1 页、2–3 张图。COLT 不跑。AISTATS 可只做 Exp.1+2。

**只实现一条主算法：** unknown-\(L_F\) + closed-form + warm rescaling（正文 Sec. 5.1）。  
对照只保留 **cold restart**。不做 Hsieh、不做隐式根、不扫四层算法。本轮对齐也不加任何新 baseline；若审稿人之后要求比较，再单独加 Hsieh。

---

## 0. 验证什么 / 不验证什么

| 要看的 | 对应 | 不要看的 |
|---|---|---|
| Self-play 下固定 comparator 的累计 regret 进入平台 | Thm 4.B | last-iterate 收敛 |
| 平均对局的 restricted gap \(\sim 1/T\) | Cor. gap | 全域 gap |
| \(Q_t^{\mathrm{obs}}\) 饱和 | movement control | 显式常数 \(\bar Q\)、32/64 是否紧 |
| 同一状态、不检测、对手切换后仍可用 | Thm 4.D | 任意对手下界 |
| 分离例上 \(V_T(u^\star)=0\)，regret 不跟 \(\sqrt T\) 走 | App. A + fallback | “真实 regret 是 \(\Omega(\sqrt T)\)” |
| \(\ell_t\) 有限次加倍后冻结；restart 轨迹不连续 | Claim 64 + Remark cold | restart 的信息论 \(\sqrt{K}\) 下界 |
| 闭式每轮 \(O(d)\) | Thm 4.A | 有限精度 bit complexity |

失败处理见 §7。图不好看就撤图，改纯理论投稿。

---

## 1. 目录

骨架已建。编码合同：`SPEC.md`。超参：`configs/default.yaml`。依赖：`requirements.txt`。

```
experiment/
  PLAN.md
  SPEC.md                 # Step 0：D005 更新顺序与不变量
  requirements.txt
  src/                    # Step 1 起写 games.py / metrics.py
  configs/default.yaml
  scripts/
  results/                # 不进 git
  figures/                # 定稿后再考虑进 git
```

依赖：`numpy`、`matplotlib`、`pyyaml`。不需要 GPU。

---

## 2. 实例

统一：无约束、光滑、凸凹。双方同时行动；每人只看自己的 \(g_t\)；unknown-\(L_F\) 在回合结束后用公开 \(z_t\) 算 \(\chi_t\)。

构造函数默认 \(a=b=0\)（体检用）。**投稿实验**统一用

\[
a=0.4\,e_1,\qquad b=0.4\,e_2,
\]

使鞍点不在原点，从而 \(w_1=0\) 不是 self-play 静止点。任何实验脚本都**禁止**覆盖 `player.action`。

**G1 双线性（平均 iterate 易好、last-iterate 可转圈）**

\[
\Phi(x,y)=(x-a)^\top A(y-b),\qquad
g^x=A(y-b),\quad g^y=-A^\top(x-a).
\]

- 主图：\(d=10\)，\(A=I\)，单次确定性运行。
- 附录：3 个 `seed` 的高斯 \(A\)，先除以 \(\|A\|_2\)（spectral normalization），同样 \(w_1=0\)、同一 \((a,b)\)。三条线分开画，不画均值带。
- 鞍点 \((a,b)\)。\(A=I\) 时 self-play 落在 \(\mathrm{span}\{e_1,e_2\}\)（二维 sanity check，不是十维）；真正高维只由附录 Gaussian 承担。
- Restricted gap：半径 \(R=1\)、**以鞍点为心** 的欧氏球。闭式  
  \(\operatorname{Gap}=R\bigl(\|A^\top(\bar x-a)\|+\|A(\bar y-b)\|\bigr)\)。

**G2 强凸强凹二次（self-play 平台最干净）**

\[
\Phi(x,y)=\frac{\mu}{2}\|x-a\|^2+(x-a)^\top A(y-b)-\frac{\mu}{2}\|y-b\|^2,
\qquad \mu=0.2,\ d=10.
\]

- \(g^x=\mu(x-a)+A(y-b)\)，\(g^y=\mu(y-b)-A^\top(x-a)\)。
- 鞍点 \((a,b)\)。Gap 在平移坐标 \(x'=\bar x-a,\,y'=\bar y-b\) 上用现有解析式，禁止网格搜。
- Exp.1 / Exp.2-left / Exp.3 共用这一份 G2。

**G3 附录分离例（非 self-play）**

\[
\Phi(x,y)=2\sqrt{1+x^2}+xy-\tfrac12 y^2,\qquad y_t\equiv 1.
\]

- \(g^x=2x/\sqrt{1+x^2}+y\)，\(g^y=y-x\)（注意 \(g^y=-\nabla_y\Phi\)）。
- Comparator：\(u^\star=-1/\sqrt{3}\)。应有 \(V_t(u^\star)=0\)，\(1\le G_t\le 3\)。

对手协议（Exp.2 用）：

| 名字 | 行为 |
|---|---|
| `self` | 对方也跑 D005 |
| `const` | \(y_t\equiv 1\)（G3）或 \(y_t\equiv e_1\)（高维 Exp.3；不是 \(b\)） |
| `slow` | 见下；绕 \(b\) 附近振荡，\(T_{\mathrm{per}}=200\) |
| `switch` | 前 \(T/2\) 为 `self`，之后切到 `slow`；切换处必须连续 |

**`slow` / `switch` 连续性（Exp.2-left 硬约束）。** 记 \(t_\star=T/2\)，\(y_{\mathrm{anchor}}=y_{t_\star}\)（最后一轮 self-play 动作）。\(t>t_\star\) 时

\[
y_t
=
y_{\mathrm{anchor}}
+\sin\Bigl(\frac{2\pi(t-t_\star)}{T_{\mathrm{per}}}\Bigr)\,e_1
=
b+(y_{\mathrm{anchor}}-b)
+\sin\Bigl(\frac{2\pi(t-t_\star)}{T_{\mathrm{per}}}\Bigr)\,e_1.
\]

这样 regime switch 是“从已收敛的 self-play 动作开始沿 \(e_1\) 慢振”，不是把 \(y\) 瞬移到原点附近的正弦。离散一步差恰好是正弦增量：

\[
\|y_{t_\star+1}-y_{t_\star}\|
=\Bigl|\sin\frac{2\pi}{T_{\mathrm{per}}}\Bigr|
\approx 0.0314.
\]

Smoke（失败则停，不要出图）：

- \(\|y_{\mathrm{anchor}}-b\|\le 0.05\)（\(T/2\) 时 self-play 已靠近鞍点；否则不是“在 \(b\) 附近振荡”）。
- \(\bigl|\|y_{t_\star+1}-y_{t_\star}\|-\lvert\sin(2\pi/T_{\mathrm{per}})\rvert\bigr|\le 10^{-12}\)（没有额外跳变）。

---

## 3. 主算法 D005（必须与正文一致）

每人独立维护一份状态。记号与 `note/sections/main-result.tex`、`adaptive.tex` 对齐。

### 3.1 初始化（算法不读 \(L_F,T,G,V,u\)）

默认超参（两边相同，写进 `configs/default.yaml`）：

```
epsilon: 1.0
beta0:   1.0
ell1:    1e-3        # 刻意偏小，让 doubling 发生
T:       20000
seeds:   [0, 1, 2]
```

\[
\beta_1=\beta_0+\frac{64\,\ell_1^2}{\beta_0},\qquad
\gamma=\epsilon\beta_1\ \text{（此后冻结）}.
\]

- \(\widehat M_1=\gamma\)，\(B_1=4\)，\(\overline V_1=4\gamma^2\)，
  \(\alpha_1=\epsilon/(\sqrt{B_1}\log^2 B_1)\)（自然对数）。
- \(\zeta_1=0\)，\(w_1=0\)，\(h_1=0\)，\(\sum_{s<1}g_s=0\)。
- \(\ell_2=\ell_1\)（第 1 轮不加倍）。

### 3.2 径向映射（数值稳定）

\[
a_t=\sqrt{\overline V_t},\quad
k_t=\beta_t\alpha_t/a_t,\quad
q_t(s)=\frac{a_t}{\beta_t}\log\frac{1+k_t e^{s/a_t}}{1+k_t}.
\]

实现用 `log1p`；\(s/a+\log k\) 大时改写成 \(s/a+\log k+\log1p(e^{-(\cdot)})\)，避免 overflow。正文已允许该等价形式。

### 3.3 出招

\[
\Theta_t=h_t+\sum_{s=1}^{t-1}g_s,\qquad
\sigma_t=(\|\Theta_t\|-\zeta_t)_+,
\]

\[
w_t=
\begin{cases}
0,&\sigma_t=0,\\
-q_t(\sigma_t)\,\Theta_t/\|\Theta_t\|,&\sigma_t>0.
\end{cases}
\]

### 3.4 回合后更新（顺序不能乱）

双方先同时出 \(w_t\)，环境返回 \(g_t\)，再更新：

1. \(\Delta_t=g_t-h_t\)。\(\Delta=0\) 则 \(\widehat\Delta=0\)，否则  
   \(\widehat\Delta=\Delta\min\{1,\widehat M_t/\|\Delta\|\}\)。
2. \(\lambda_t=2\|\widehat\Delta_t\|^2/a_t\)（此处 \(a_t\) 仍是本轮出招用的旧值）。
3. 再更新 clipping：  
   \(\widehat M_{t+1}=\max\{\widehat M_t,\|\Delta_t\}\)，  
   \(B_{t+1}=4+\sum_{s\le t}\|\widehat\Delta_s\|^2/\widehat M_s^2\)，  
   \(\overline V_{t+1}=4\widehat M_{t+1}^2+\sum_{s\le t}\|\widehat\Delta_s\|^2\)，  
   \(\alpha_{t+1}=\epsilon/(\sqrt{B_{t+1}}\log^2 B_{t+1})\)。
4. \(\zeta_{t+1}=\zeta_t+\lambda_t\)。
5. \(t\ge 2\) 时用公开 \(z_t,z_{t-1}\) 与自己的 \(g_t,g_{t-1}\)：
   \[
   \chi_t=\begin{cases}
   \|g_t-g_{t-1}\|/\|z_t-z_{t-1}\|& z_t\neq z_{t-1},\\
   0& \text{otherwise.}
   \end{cases}
   \]
   \(\chi_t>\ell_t\) 则 \(\ell_{t+1}=2\max\{\ell_t,\chi_t\}\)，否则不变。  
   \(\beta_{t+1}=\beta_0+64\ell_{t+1}^2/\beta_0\)。新 \(\beta\) **只用于下一轮**。
6. \(h_{t+1}=g_t\)，累加器 \(\leftarrow\) 累加器 \(+g_t\)。
7. 用新状态算 \(w_{t+1}\)。

**Warm 永不重置** FTRL 累加器、clipping、\(\zeta\)、动作。

### 3.5 Cold restart（唯一对照）

与 D005 相同的 \(\chi,\ell,\beta\) 规则；一旦 \(\chi_t>\ell_t\)，除 \(\ell,\beta,\epsilon\) 外全部清零：

- \(w\leftarrow 0\)，累加器、\(h\)、\(\zeta\) 清零；
- clipping 按**新的** \(\beta_+\) 重初始化：\(\gamma_+\leftarrow\epsilon\beta_+\)，\(\widehat M=\gamma_+\)，\(B=4\)，\(\overline V=4\gamma_+^2\)，\(\alpha\) 重算。

这对应 Remark cold，不是主算法。图注写明。

---

## 4. 度量（每轮记一份）

对玩家 \(i\in\{x,y\}\)：

- \(\operatorname{Reg}_t^i(u_i)=\sum_{s\le t}(\Phi\text{ 差})\)。G1/G2：**禁止**传入 `u_x`/`u_y` 覆盖；`RunningMetrics` 必须用 `game.saddle()`（即 \((a,b)\)），跑完后 `assert allclose(u_x,a)` 与 `allclose(u_y,b)`。G3 是唯一例外：X 用 \(u^\star=-1/\sqrt{3}\)，不是鞍点。图例 G1/G2 写 \(\mathrm{Reg}^x(a),\mathrm{Reg}^y(b)\)，G3 写 \(\mathrm{Reg}^x(u^\star)\)；禁止残留 \(\mathrm{Reg}^x(0)\)。
- \(\operatorname{LinReg}_t^i(u_i)=\sum_{s\le t}\langle g_s^i,w_s^i-u_i\rangle\)。
- \(Q_t=\sum_{s<t}\|z_{s+1}-z_s\|^2\)。
- \(G_t^i=\max_{s\le t}\|g_s^i\|\)。
- \(V_t^i(u_i)\)：按正文，只依赖对手轨迹与 comparator。
- 平均对局 \((\bar x_t,\bar y_t)\) 的 \(\operatorname{Gap}_{P_x,P_y}\)。
- \(\ell_t,\beta_t\)、加倍次数 \(J_t\)。
- 可选：每轮墙钟（Exp.3 不需要；runtime 表才用）。

实现上用增量更新，不要每轮从头求和 \(\Phi\)。

---

## 5. 三个实验

默认 \(T=2\times 10^4\)。主图 \(A=I\) **单次运行**（确定性，不画均值带）。高斯 \(A\) 的 3 个 seed 只进附录，spectral-normalized，三条线分开画。**禁止** off-saddle probe：第 1 轮必须是算法给出的 \(w_1=0\)。

### Exp.1 Self-play 常值 regret（主文 Figure 1）

- 游戏：shifted G1、G2，\(A=I\)，\((a,b)=(0.4 e_1,0.4 e_2)\)。
- 协议：双方 D005，\(w_1=0\)。
- 图（一行三列）：累计 \(\operatorname{Reg}_t^x(a),\operatorname{Reg}_t^y(b)\)；log-log restricted gap（加斜率 \(-1\) 参考，正文写 consistent with \(O(1/T)\)，不写 “decays at the \(1/T\) rate”）；\(Q_t\)。
- 可加一条 \(\sqrt t\) 虚线，**不要**写成 SOTA 对比。
- Smoke：\(\|x_1\|=\|y_1\|=0\)；\(t=2\) 已离开原点；`J>=1` 且后半段冻结。
- **成功：** 后半段 regret 与 \(Q\) 近似水平；gap 随 \(T\) 下降。G1 last-iterate 打转可忽略。
- **失败且可修：** \(T\) 不够、\(\texttt{ell1}\) 过大/过小、看了 \(\|z_t\|\) 而非平均 gap。
- **失败且撤图：** 对鞍点的 regret 持续按 \(\sqrt T\) 涨，且查过实现仍如此 → 停，查证明/代码，不要投稿该图。

### Exp.2 Same-run 切换 + 分离例（主文 Figure 2）

- **2a 切换：** shifted G2 上 `switch`（前半 self-play，后半 §2 的连续 `slow`）。算法不检测、不换超参、不重启。竖线标 \(T/2\)。Claim 只讲 same-state / no reset / 数值稳定；**禁止**写 “unchanged in shape” 或“验证了 fallback rate”。
- **2b 分离例：** G3 + `const`，**不改、不重跑**。画 \(\operatorname{Reg}_t^x(u^\star)\)，旁注 \(V_t(u^\star)=0\)。不要把该曲线解释成别人 \(\Omega(\sqrt T)\) 的下界。
- **成功：** 一套状态跨过切换点，且切换处 \(\|y\|\) 无 \(O(1)\) 跳变；2b 的 \(V\equiv 0\) 且 regret 无明显 \(\sqrt T\) 斜向上。
- **失败且可修：** 切换后短时振荡（可接受）；2b 的 \(V\) 因数值不是精确 0（应用解析梯度，应精确 0）。
- **失败且撤图：** 切换后 regret 爆炸且实现无误 → 与 same-run 叙事冲突，该图不进主文。

### Exp.3 Warm vs restart（主文 Figure 3 或表）

- 游戏：同一份 shifted G2，\(\texttt{ell1}=10^{-3}\)（必须真的加倍）。
- 两条学习器其余相同，只改是否 reset。
- **协议：** 主图用 X vs \(y\equiv e_1\)（不是 \(b\)）。诱导损失的最小点在 \(x^\star=a-A(e_1-b)/\mu\)；\(A=I\) 时 \(x^\star=-4.6 e_1+2 e_2\)，\(\|x^\star\|\approx 5.02\)。Cold restart 仍跳回**原点**（算法初值），不是鞍点。
- 图（一行三列，前 400 步）：\(J_t\)（有限次后冻结）；\(\|x_t\|\)（restart 在 \(t=3\) 回原点）；\(\operatorname{Reg}^x(a)\)（一次性滞后）。
- **成功：** \(J=1\) 后冻结；restart 的原点跳跃可见。不把终值相差写成 \(\sqrt{K}\) 税。正文写 cold-restart “faces the same game and opponent sequence”，禁止 “same realized play”。
- **失败且可修：** 从未加倍 → 再减小 \(\ell_1\) 或加大 \(A,\mu\)。
- **失败且撤图：** 连 vs-const 的前段也看不出 reset。正文本来就不证明 restart 必更差；删图，不改定理表述。

### 不做（除非附录还有空）

闭式 vs 隐式 runtime。主贡献不是速度。若做：\(d\in\{10,50,200\}\) 一张表，主文一句话。

---

## 6. 落地步骤

按顺序，前一步通了再往下。每步在 `results/` 留一个 json 摘要（最终 \(\operatorname{Reg},Q,J,V\)）。

**Step 0.（已完成）** `SPEC.md` + `configs/default.yaml` + 目录 + 依赖。

**Step 1–2.（已完成）** `src/games.py`，`src/metrics.py`。检查：`python experiment/scripts/check_games_metrics.py`。

**Step 3 前半.（已完成）** 冻结 \(\beta\) 闭式核：`src/learner.py`。检查：`python experiment/scripts/check_frozen_beta.py`。G2、\(d=1\)、\(T=2000\)、\(\beta=5L_F\)。原点 self-play 是静止点，故体检用首步 \((1,-0.5)\)。摘要：`results/frozen_beta_g2.json`。

**Step 3 后半.（已完成）** 打开 doubling。检查：`python experiment/scripts/check_doubling.py`。\(t=1\) 不加倍；\(\chi\) 只用自己的 \(g\)；\(\beta\) 非降；\(J\le\lceil\log_2(L^{\mathrm{row}}/\ell_1)\rceil\)；\(\gamma\) 与 \(G_{\mathrm{cum}}\) 不重置。摘要：`results/doubling_g2.json`。下一步 Exp.1，不要跳。

**Step 4.（已完成，并入 Exp.1 / 对拍）** 闭式 vs 隐式半径：`python experiment/scripts/check_closedform.py`。全程 `assert_invariants()`（\(\beta\) 非降、\(\gamma\) 冻结、\(G_{\mathrm{cum}}\) 只累加、\(B\ge 4\)）。

**Step 5.（已完成）** Exp.1：`python experiment/scripts/exp1_selfplay.py` 写 json；`python experiment/scripts/plot_exp1.py --appendix` 出图。主图 `figures/exp1_selfplay.pdf`（\(A=I\) 单跑）。高斯 \(A\) 三 seed 进 `figures/exp1_G*_gaussian.pdf`，不画均值带。下一步 Exp.2，cold restart 仍推迟。

**Step 6.（已完成）** Exp.2：`run_loop` 支持外生 Y；`python experiment/scripts/exp2_bobw.py`；`python experiment/scripts/plot_exp2.py`。2b 先跑 G3+\(y\equiv 1\)（\(V=0\) 精确成立）；2a 为 G2 在 \(T/2\) 切到 `slow`。Cold restart 仍推迟。

**Step 7.（已完成）** Exp.3：`python experiment/scripts/check_restart.py`；`python experiment/scripts/exp3_restart.py`；`python experiment/scripts/plot_exp3.py`。主图 `figures/exp3_restart.pdf`（G2 vs \(y\equiv e_1\)，前 400 步）。Self-play json 保留但不进主图。

**Step 8.（已完成）** 出图。三个 plot 脚本统一 Okabe-Ito 色盲调色板、pdf fonttype 42 矢量字体、线型可分（warm 实线 / restart 虚线）。Exp.3 第三面板改全地平线 + 前 400 轮 inset，标题去掉 \(\sqrt K\) 表述。全链路重跑：json 摘要逐位复现，8 张 pdf/png 定稿于 `figures/`。图注（现象 + 定理编号）在 LaTeX 侧待写，不写“优于 Hsieh”。

**Step 9.（已完成，将被 Step 10 作废图替换）** Go / no-go（见 §7）。裁决：Exp.1–2 进主文；Exp.3 进主文，只讲有限次加倍 + 一次断状态，不声称 restart 终值更差或 \(\sqrt{K}\) 税。

**Step 10.（待执行：投稿前对齐）** 作废旧 `results/exp1_*.json`、`exp2_G2_switch.json`、`exp3_*.json` 及对应主图。G3 json 保留。按顺序：

1. `games.py`：G1/G2 平移；gap 以鞍点为心；`gaussian(..., normalize=True)`。
2. 体检去掉 probe；保留 \(a=b=0\) 时“原点是静止点”的反面测试。
3. 实验脚本：禁止覆盖 `action`；G1/G2 对 `game.saddle()` 做 comparator 断言；Exp.2-left 用连续 `slow`。
4. 重跑 Exp.1 全套、Exp.2 `--only 2a`、Exp.3；出图；改 `experiments.tex` + 附录 Gaussian 图。
5. 新图再走 §7。不混用新旧 json。

---

## 7. Go / no-go

对每张图只判三类：

1. **进主文：** 现象与定理同形。
2. **进 appendix / 不提：** 能跑但不加分（例如 G1 的 last-iterate 很吵）。
3. **整节删除：** 对准定理的量明显反了，或对照（restart）没有故事。

| 现象 | 动作 |
|---|---|
| 平台期晚、常数大 | 加大 \(T\)；仍水平就进文，不拟合 \(\bar Q\) |
| last-iterate 差、平均 gap 好 | 只画平均；与“不声称 last-iterate”一致 |
| restart 不更差（vs-const 前段也无跳跃） | 删 Exp.3 |
| self-play regret 真按 \(\sqrt T\) 涨 | 先查 §3.4 顺序与 \(g^y\) 符号；仍反则停实验、查证明 |
| 切换后崩溃 | 不进主文；论文改回纯理论 |

**原则：** 失败的图比没有图更伤。COLT 直接跳过本文件夹。

---

## 8. 主文怎么引用

放在 Main Theorem 与 limitations 之间。

- Figure 1 = Exp.1（G2 为主，G1 作补充可进 appendix）。
- Figure 2 = Exp.2a + 2b。
- Figure 3 = Exp.3（G2 vs \(y\equiv e_1\)，前 400 步；self-play 无可见跳跃，不进主图）。

正文三句话就够：self-play 饱和；同一轨迹切换后不重开（不检测、不重置，不声称 shape 不变）；warm 有限次加倍、restart 会断状态。不把实验写成贡献条目。附录给 spectral-normalized Gaussian 的 Exp.1。

---

## 9. 与正文公式的索引

| 实现对象 | 出处 |
|---|---|
| 反馈 \(g^x=\nabla_x\Phi,\ g^y=-\nabla_y\Phi\) | `setting.tex` |
| last-gradient \(h_1=0,h_t=g_{t-1}\) | `eq:hint` |
| clipping \(B,\overline V,\alpha\) | `eq:clip`, `eq:B-V` |
| \(q_t,w_t,\zeta\) | `eq:q-t-var`, `eq:w-cf`, `eq:zeta` |
| \(\chi_t\)、加倍、系数 64 | `eq:chi`, `eq:ell-update`, `eq:t006-1` |
| regret / \(V_T\) / restricted gap | `eq:regret`, `eq:VT`, `eq:restricted-gap` |
| 分离例 | `appendix_a.tex` / `note/example.md` |
| cold restart | `remark:cold` |
