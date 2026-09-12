# 实验方案（flagship = `thm:t006` / D005）

编码合同：`SPEC.md`。评价与完成勾选：`reflection.txt`。执行顺序：本文 §6，**Step 11–18 全部必做**，做完后 `reflection.txt` 第 5 节每一条都必须落实。

**主算法：** unknown-\(L_F\) + closed-form + warm rescaling（正文 Sec. 5.1 / D005）。  
**Literature baseline（必做）：** Hsieh et al. (2021)，`src/hsieh.py`。  
**机制 ablation：** cold restart（Exp.3）。Restart **不是** literature baseline，不能代替 Hsieh。

对准 ICLR / ICML 主文约 1–2 页加附录表。COLT 不跑。

---

## 0. 验证什么 / 不验证什么

| 要看的 | 对应 | 不要看的 |
|---|---|---|
| Self-play 下固定 comparator 的累计 regret 进入平台 | Thm.t006.B | last-iterate 收敛 |
| 平均对局的 restricted gap \(\sim 1/T\) | Cor. gap | 全域 gap |
| \(Q_t^{\mathrm{obs}}\) 饱和 | movement control | 显式常数 \(\bar Q\)、32/64 是否紧 |
| 同一状态、不检测、对手切换后仍可用 | Thm.t006.D | 把负 \(\mathrm{Reg}^x(a)\) 当成 fallback 验证 |
| 分离例上 \(V_T(u^\star)=0\)，D005 regret 不跟 \(\sqrt T\) 走 | App. A + fallback | 在未跑 Hsieh 前声称别人 \(\Omega(\sqrt T)\) |
| D005 vs Hsieh 在 G3 / switch 上的**真实** regret | BoBW 经验对照 | 用 cold restart 冒充 literature baseline |
| \(\ell_t\) 有限次加倍后冻结；restart 轨迹不连续 | Claim 64 + Remark cold | restart 的信息论 \(\sqrt{K}\) 下界 |
| doubling 次数 / 终值 \(\ell,\beta\) 随 \(\|A\|\) 变化 | unknown-\(L_F\) | 把谱归一化 Gaussian 当成 \(L_F\) sweep |
| regret 随 \(\sqrt{V_T(u)}\) 变化；非平稳且 \(V(u^\star)=O(1)\) | comparator-local 适应 | 只保留 \(V=0\) 端点 |
| 多 horizon 终端统计 | \(O(1)\)、\(O(1/T)\) | 只靠单条曲线目测斜率 |
| 闭式每轮 \(O(d)\) | Thm.t006.A | 有限精度 bit complexity；runtime 表不是本轮必做 |

失败处理见 §7。对照图失败则先修实现，禁止带着坏 Hsieh 投稿。

---

## 1. 目录

骨架已建。编码合同：`SPEC.md`。超参：`configs/default.yaml`。依赖：`requirements.txt`。

```
experiment/
  PLAN.md                 # 本文件：实验设计 + 必做落地步骤
  SPEC.md                 # D005 / Hsieh / comparator 合同
  reflection.txt          # 评价；第 5 节条目必须由 §6 全部落实
  README.md               # Step 14 补
  requirements.txt
  src/                    # games.py / metrics.py / learner.py / hsieh.py
  configs/default.yaml
  scripts/
  results/                # compact summary json + npz，不进 git
  figures/
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

### 3.5 Cold restart（机制 ablation，不是 literature baseline）

与 D005 相同的 \(\chi,\ell,\beta\) 规则；一旦 \(\chi_t>\ell_t\)，除 \(\ell,\beta,\epsilon\) 外全部清零：

- \(w\leftarrow 0\)，累加器、\(h\)、\(\zeta\) 清零；
- clipping 按**新的** \(\beta_+\) 重初始化：\(\gamma_+\leftarrow\epsilon\beta_+\)，\(\widehat M=\gamma_+\)，\(B=4\)，\(\overline V=4\gamma_+^2\)，\(\alpha\) 重算。

这对应 Remark cold，不是主算法，也不是 Hsieh。图注写明。

### 3.6 Hsieh (2021)（literature baseline，必做）

见 `SPEC.md`「Hsieh (2021) literature baseline」。`src/hsieh.py` 独立实现。主对照 G3 与 G2 switch；G2 self-play 只作实现体检。

---

## 4. 度量（每轮记一份）

对玩家 \(i\in\{x,y\}\)：

- \(\operatorname{Reg}_t^i(u_i)=\sum_{s\le t}(\Phi\text{ 差})\)。Comparator 按 `SPEC.md` 协议第 7 条：
  - G1/G2 **self-play**：必须 `game.saddle()`，跑完断言 \(u_x=a\)、\(u_y=b\)。图例 \(\mathrm{Reg}^x(a),\mathrm{Reg}^y(b)\)。
  - G2 **vs-const**：必须 \(u_x=x^\star=a-A(e_1-b)/\mu\)（`induced_minimizer_x`），断言 \(V_x(x^\star)=0\)。图例 \(\mathrm{Reg}^x(x^\star)\)。
  - G3：\(u^\star=-1/\sqrt{3}\)。图例 \(\mathrm{Reg}^x(u^\star)\)。
  - 禁止残留 \(\mathrm{Reg}^x(0)\)。禁止把 self-play 的鞍点断言套到 vs-const / Hsieh-G3 上。
- \(\operatorname{LinReg}_t^i(u_i)=\sum_{s\le t}\langle g_s^i,w_s^i-u_i\rangle\)。
- \(Q_t=\sum_{s<t}\|z_{s+1}-z_s\|^2\)。
- \(G_t^i=\max_{s\le t}\|g_s^i\|\)。
- \(V_t^i(u_i)\)：按正文，只依赖对手轨迹与 comparator。
- 平均对局 \((\bar x_t,\bar y_t)\) 的 \(\operatorname{Gap}_{P_x,P_y}\)。
- \(\ell_t,\beta_t\)、加倍次数 \(J_t\)。
- 可选：每轮墙钟（Exp.3 不需要；runtime 表才用）。

实现上用增量更新，不要每轮从头求和 \(\Phi\)。

---

## 5. 实验（含已完成的三张主图 + 必做增补）

默认 \(T=2\times 10^4\)。主图 \(A=I\) **单次运行**（确定性，不画均值带）。高斯 \(A\) 的 3 个 seed 只进附录，spectral-normalized，三条线分开画。**禁止** off-saddle probe：第 1 轮必须是算法给出的 \(w_1=0\)。新 run 只写 compact summary + npz。

### Exp.1 Self-play 常值 regret（主文 Figure 1；已完成，本轮不改协议）

- 游戏：shifted G1、G2，\(A=I\)，\((a,b)=(0.4 e_1,0.4 e_2)\)。
- 协议：双方 D005，\(w_1=0\)，comparator = 鞍点。
- 图（一行三列）：\(\operatorname{Reg}_t^x(a),\operatorname{Reg}_t^y(b)\)；log-log restricted gap；\(Q_t\)。G2 主图 \(T=2\times10^4\)。G1 行 \(T=2\times10^5\)。
- G1 + \(A=I\) 的 regret 恒为 0 是结构退化；非平凡 G1 regret 在 Gaussian 附录。
- 可加 \(\sqrt t\) 虚线，**不要**把这条虚线写成 SOTA。Hsieh 的 self-play 曲线若画，只进 appendix 体检，不进 Figure 1 主文当 BoBW。
- Smoke：\(\|x_1\|=\|y_1\|=0\)；\(t=2\) 已离开原点；`J>=1` 且后半段冻结。
- **成功 / 失败：** 同原判据。对鞍点 regret 按 \(\sqrt T\) 涨且实现无误 → 停，查证明。

### Exp.2 Same-run 切换 + 分离例（主文 Figure 2；Step 12 / 13 必须改）

- **2a 切换：** 协议不变（连续 `slow`，smoke 不变）。**主曲线改为** \(\|x_t-a\|\)、\(\|y_t-b\|\) 与 \(V_t^x(a)\)。\(\mathrm{Reg}^x(a)\) 只许 inset / appendix。Claim 只讲 same-state / no reset；**禁止**写 unchanged in shape 或「验证了 fallback rate」。
- **2b 分离例 + Hsieh：** G3 + \(y\equiv 1\)。D005 的 \(\mathrm{Reg}^x(u^\star)\) 保留；Step 13 **必须**叠加 Hsieh 同一 comparator。不要把 D005 单曲线解释成别人 \(\Omega(\sqrt T)\) 的下界；有了 Hsieh 真实曲线之后，按实际形状写（饱和 vs \(\sqrt{T}\)，或双方饱和则写入 limitations）。
- **成功：** 切换处 \(\|y\|\) 无 \(O(1)\) 跳；\(V\) 在竖线后升起；\(\gamma\) 不重置；2b 的 \(V\equiv 0\)；Hsieh 实现通过 G2 self-play 平台体检。
- **失败且撤图：** 切换后爆炸；或 Hsieh 在 G2 self-play 上按 \(\sqrt{T}\) 涨（baseline 作废，先修实现）。

### Exp.3 Warm vs restart（主文 Figure 3；Step 11 必须改 comparator）

- 游戏：shifted G2，\(\texttt{ell1}=10^{-3}\)。只改是否 reset。
- **协议：** X vs \(y\equiv e_1\)。Comparator **必须**是 \(x^\star=a-A(e_1-b)/\mu\)（\(A=I\) 时 \(\approx -4.6 e_1+2 e_2\)，\(\|x^\star\|\approx 5.02\)），**禁止**再对鞍点 \(a\) 画主 regret。Cold restart 仍跳回原点。
- 图：\(J_t\)；\(\|x_t\|\)（可加水平线 \(\|x^\star\|\)）；\(\operatorname{Reg}^x(x^\star)\) 应饱和，warm / restart 常数偏移。
- **成功：** \(J=1\) 后冻结；原点跳跃可见；对 \(x^\star\) 后半段水平。不把终值差写成 \(\sqrt{K}\) 税。禁止 “same realized play”。
- **失败且撤图：** 对 \(x^\star\) 仍线性变负；或看不出 reset。

### Exp.Hsieh（Step 13，必做）

- `scripts/exp_hsieh.py` + `plot_hsieh.py`。
- 必跑：G3 const；G2 switch（与 Exp.2a 同一对手序列）；G2 self-play 体检。
- 主文：G3 对照进 Figure 2 右（或单独对照图）。一旦出现第二条算法曲线，删掉 “no other algorithm is shown”。

### Exp.horizon（Step 15，必做）

从已有 hist / G1 长跑抽取终端表，原则上不重跑：

\[
T\in\{5\times 10^3,\,10^4,\,2\times 10^4\}\ \text{(G2)},\qquad
T\in\{2\times 10^4,\,10^5,\,2\times 10^5\}\ \text{(G1 \(Q\))}.
\]

列：\(\mathrm{Reg}^x(a)\)、\(Q_T\)、\(\mathrm{Gap}_T\)、\(T\cdot\mathrm{Gap}_T\)、\(J\)。进 appendix。不要跑 \(T<5\times 10^3\) 凑数。

### Exp.LF（Step 16，必做）

G2 self-play，\(A=cI\)，\(c\in\{0.25,0.5,1,2,4,8\}\)，**禁止**谱归一化。记录 \(J,\ell_T,\beta_T,\mathrm{Reg}_T(a),\mathrm{Gap}_T,Q_T\)。进 appendix。正文至多一句 finite doubling tracks realized smoothness。

### Exp.VT（Step 17，两条都必做）

- **17a 幅度扫频：** \(y_t=b+\eta\sin(2\pi t/T_{\mathrm{per}})e_1\)，\(\eta\) 数档。Comparator 用诱导最优固定点。终值 regret vs \(\sqrt{V_T(u)}\)。
- **17b 更强分离：** 非平稳对手、观测 variation 大、但 \(V_T(u^\star)=O(1)\)。D005 应饱和。不宣称旧算法 \(\Omega(\sqrt{T})\)；Hsieh 可同图画（Step 13 之后）。

### 明确不做

- 闭式 vs 隐式 runtime / \(d\) 扫描。
- 四层算法全扫。
- 把 restart 终值差写成 \(\sqrt{K}\) 税。
- 把 Exp.2 改成不连续大跳变（破坏 part D）。
- 用任何自制算法代替 Hsieh 当 literature baseline。

---

## 6. 落地步骤

按顺序执行。Step 0–10 已完成。**Step 11–18 全部必做**，没有「可选 / 视需要 / 最小闭环」。每步过 go/no-go 再往下。产物：compact `summary` json + 下采样 npz（禁止巨型全程 hist json）。

**Step 0–10.（已完成）** D005 实现、shifted saddle、\(w_1=0\)、连续 switch、Gaussian 附录、cold restart 图、`experiments.tex` 弱声称插图。Step 3 冻结-\(\beta\) 体检用平移鞍点，不再用手工改第一步。**本阶段未加 Hsieh，这是缺口，由 Step 13 补上，不是最终状态。**

**Step 11. Exp.3 comparator \(\to x^\star\)**（落实：vs-const 不再对鞍点累计）

- `QuadraticGame.induced_minimizer_x(y)`；`run_vs_const` 把 \(x^\star\) 传入 `RunningMetrics`。
- Smoke：`V_x(x^\star)==0`；`J>=1`；restart \(t=3\) 回原点；warm 不回；\(\mathrm{Reg}^x(x^\star)\) 后半段水平。
- 重跑 `check_restart.py`、`exp3_restart.py`、`plot_exp3.py`。右图例 \(\mathrm{Reg}^x(x^\star)\)。
- 改 `experiments.tex` Figure 3 图注。仍不声称 \(\sqrt{K}\) 税。
- **Go：** 两曲线饱和且差一个常数。**No-go：** 对 \(x^\star\) 仍 \(\Theta(T)\) 变负。

**Step 12. Exp.2-left 换纵轴**（落实：负 regret 不作主曲线）

- 协议不变。主曲线 \(\|x_t-a\|\)、\(\|y_t-b\|\)、\(V_t^x(a)\)。
- 重跑 `exp2_bobw.py --only 2a`、`plot_exp2.py`。改 Figure 2 左图注（part D）。
- **Go：** 正弦增量；距离有界；\(V\) 在竖线后升起；\(\gamma\) 不变。

**Step 13. 建立 Hsieh (2021) baseline**（落实：literature baseline）

- `src/hsieh.py`、`scripts/exp_hsieh.py`、`scripts/plot_hsieh.py`、`scripts/check_hsieh.py`（G2 self-play 平台）。合同见 `SPEC.md`。
- 必跑 G3 const、G2 switch、G2 self-play 体检。G3 对照进主文 Figure 2 右或单独图。
- 删掉正文 “no other algorithm is shown”。禁止对着 D005 调参。禁止用 restart 冒充。
- **Go：** Hsieh 在 G2 self-play 平台；G3 两条曲线可解释。**No-go：** self-play 上 Hsieh 走 \(\sqrt{T}\) → 先修实现。若 G3 上 Hsieh 也饱和，写入 limitations，仍算本步完成（实现可信且结果如实报告）。

**Step 14. README + compact 摘要**（落实：复现性）

- 新 run 一律 compact json + npz。
- 写 `experiment/README.md`：环境、全部 `check_*.py`、`exp*.py`、`plot_*.py`、`exp_hsieh.py` 的命令顺序，以及一键顺序（11→18）。
- 外层是否跟踪 `experiment/` 在 README 里写明现状。

**Step 15. Multi-horizon 表**（落实：rate 不只靠目测斜率）

- 按 §5 Exp.horizon 抽表，进 appendix（`appendix_experiments.tex`）。
- G2 的 \(\mathrm{Reg},Q\) 应不随 \(T\) 涨；\(T\cdot\mathrm{Gap}_T\) 近似常值。

**Step 16. \(L_F\) sweep**（落实：unknown-\(L_F\) 系统验证）

- 按 §5 Exp.LF。`scripts/exp_lf_sweep.py`。禁止谱归一化。
- **Go：** \(J\) 有限非降；\(c\) 增大时终值 \(\ell,\beta\) 不减；self-play 仍平台。

**Step 17. \(V_T\) 适应（两条都做）**（落实：不只 \(V=0\) 端点）

- **17a** 幅度扫频 + regret vs \(\sqrt{V_T(u)}\)。`scripts/exp_vt_sweep.py`。
- **17b** 非平稳、观测 variation 大、\(V_T(u^\star)=O(1)\)。可与 Hsieh 同图。
- **Go：** 17a 随 \(\sqrt{V}\) 平滑变差；17b 上 D005 饱和。

**Step 18. 正文与附录收口**（落实：新建议全部写进论文实验节）

改 `note/sections/experiments.tex` 与 `appendix_experiments.tex`：

- Figure 1 保留 self-play（可加 horizon 表引用）。
- Figure 2：左 = Step 12 纵轴；右 = D005 vs Hsieh on G3。
- Figure 3：\(\mathrm{Reg}^x(x^\star)\)。
- 附录：Gaussian（已有）、horizon 表、\(L_F\) sweep、\(V_T\) 17a/17b、Hsieh self-play 体检。
- 删除 “no other algorithm is shown” 以及任何「本轮不加 baseline」。
- 有 Hsieh 真实曲线且方向正确后，才允许写经验 fallback advantage；否则只写 theorem 分离 + 如实报告。

**完成判据（必须同时满足，对应 `reflection.txt` 第 5 节）**

| 新建议条目 | 由哪一步落实 |
|---|---|
| Exp.3 用诱导最优 \(x^\star\) | 11 + 18 |
| Exp.2-left 换纵轴，去掉大负 regret 主曲线 | 12 + 18 |
| Hsieh (2021) literature baseline（G3 主对照，switch 次之） | 13 + 18 |
| README / compact summary / 可复现入口 | 14 |
| multi-horizon 终端表 | 15 + 18 |
| \(L_F\) sweep（非谱归一化） | 16 + 18 |
| \(V_T\) 中间制度 | 17a + 18 |
| 非平稳且 \(V(u^\star)=O(1)\) 的更强分离 | 17b + 18 |

未勾满上表，不得声称「实验闭环完成」。

---

## 7. Go / no-go

对每张图只判三类：进主文 / 进 appendix / 整节删除。额外：

| 现象 | 动作 |
|---|---|
| 平台期晚、常数大 | 加大 \(T\)；仍水平就进文，不拟合 \(\bar Q\) |
| last-iterate 差、平均 gap 好 | 只画平均 |
| restart 对 \(x^\star\) 看不出跳跃 | 删 Exp.3 主图，不改定理 |
| self-play regret 真按 \(\sqrt T\) 涨 | 先查更新顺序与 \(g^y\)；仍反则停 |
| 切换后崩溃 | 该图不进主文 |
| Hsieh 在 G2 self-play 不平台 | **对照作废**，先修 `hsieh.py`，禁止投稿对照图 |
| G3 上 Hsieh 也饱和 | 如实写 limitations，不改口 realized-regret 分离 |
| \(L_F\) sweep 大 \(c\) 上 D005 走 \(\sqrt T\) | 该表不进文，先查 \(\chi\) / 证明 |

原则：失败的对照图比没有对照更伤。COLT 跳过本文件夹。

---

## 8. 主文怎么引用（Step 18 完成后的目标态）

放在 Main Theorem 与 limitations 之间。

- Figure 1 = Exp.1 self-play（D005）。
- Figure 2 左 = Exp.2a 距离 / \(V_t\)（part D）；右 = G3 上 D005 vs Hsieh（part C + 算法对照）。
- Figure 3 = Exp.3 warm vs restart，\(\mathrm{Reg}^x(x^\star)\)。
- 附录 = Gaussian Exp.1、horizon 表、\(L_F\) sweep、\(V_T\) 17a/17b、Hsieh self-play 体检。

正文必须覆盖：self-play 饱和；same-run 不重置；有限加倍与一次断状态；**G3 上与 Hsieh 的真实 regret 对照**；unknown-\(L_F\) 与 \(V_T\) 适应指向附录。不把 cold restart 写成 literature baseline。删除 “no other algorithm is shown”。

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
| 分离例 | `appendix_a.tex` |
| cold restart | `remark:cold` |
| Hsieh baseline | `hsieh2021adaptive`；实现合同 `SPEC.md` |
