# 实验方案（flagship = `thm:t006` / D005）

实验是插图，不是主结果。对准 ICLR / ICML 主文约 1 页、2–3 张图。COLT 不跑。AISTATS 可只做 Exp.1+2。

**只实现一条主算法：** unknown-\(L_F\) + closed-form + warm rescaling（正文 Sec. 5.1）。  
对照只保留 **cold restart**。不做 Hsieh、不做隐式根、不扫四层算法。

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

**G1 双线性（平均 iterate 易好、last-iterate 可转圈）**

\[
\Phi(x,y)=x^\top A y,\qquad
g^x=Ay,\quad g^y=-A^\top x.
\]

- 默认：\(d=10\)，\(A=I\)（再加一组 `seed` 下的高斯 \(A\) 作附录）。
- 鞍点 \((0,0)\)。
- Restricted gap 测试集：半径 \(R=1\) 的欧氏球。对平均对局有闭式  
  \(\operatorname{Gap}=R\bigl(\|A^\top \bar x\|+\|A\bar y\|\bigr)\)。

**G2 强凸强凹二次（self-play 平台最干净）**

\[
\Phi(x,y)=\frac{\mu}{2}\|x\|^2+x^\top A y-\frac{\mu}{2}\|y\|^2,
\qquad \mu=0.2,\ A=I,\ d=10.
\]

- \(g^x=\mu x+Ay\)，\(g^y=\mu y-A^\top x\)。
- 鞍点 \((0,0)\)。Gap 无闭式时，在球面上用 \(u=-\,R\,\bar y/\|\bar y\|\) 等一阶条件，或对二次型直接算（实现时写解析式，不要网格搜）。

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
| `const` | \(y_t\equiv 1\)（G3）或 \(y_t\equiv e_1\)（高维） |
| `slow` | \(y_t=\sin(2\pi t/T_{\mathrm{per}})\,e_1\)，\(T_{\mathrm{per}}=200\) |
| `switch` | 前 \(T/2\) 为 `self`，之后切到 `const` 或 `slow` |

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

- \(\operatorname{Reg}_t^i(u_i)=\sum_{s\le t}(\Phi\text{ 差})\)，默认 \(u_i=\) 鞍点（G3 的 X 用 \(u^\star\)）。
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

默认 \(T=2\times 10^4\)。主图 \(A=I\) **单次运行**（确定性，不画均值带）。高斯 \(A\) 的 3 个 seed 只进附录，三条线分开画。原点是鞍点静止点，故首步用固定偏离 \(e_1/-e_1\)（不是算法声称的 \(w_1=0\) 行为）。

### Exp.1 Self-play 常值 regret（主文 Figure 1）

- 游戏：G1、G2。
- 协议：双方 D005。
- 图（一行三列）：累计 \(\operatorname{Reg}_t^i(0)\)；log-log restricted gap（加斜率 \(-1\) 参考）；\(Q_t\)。
- 可加一条 \(\sqrt t\) 虚线，**不要**写成 SOTA 对比。
- **成功：** 后半段 regret 与 \(Q\) 近似水平；gap 随 \(T\) 下降。G1 last-iterate 打转可忽略。
- **失败且可修：** \(T\) 不够、\(\texttt{ell1}\) 过大/过小、看了 \(\|z_t\|\) 而非平均 gap。
- **失败且撤图：** 对鞍点的 regret 持续按 \(\sqrt T\) 涨，且查过实现仍如此 → 停，查证明/代码，不要投稿该图。

### Exp.2 Same-run 切换 + 分离例（主文 Figure 2，优先做）

- **2a 切换：** G2 上 `switch`（前半 self-play，后半 `slow` 或 `const`）。算法不检测、不换超参、不重启。竖线标 \(T/2\)。
- **2b 分离例：** G3 + `const`。画 \(\operatorname{Reg}_t^x(u^\star)\)，旁注 \(V_t(u^\star)=0\)。不要把该曲线解释成别人 \(\Omega(\sqrt T)\) 的下界；只说 fallback 上界去掉 \(\sqrt T\) 主阶，曲线与饱和/缓慢增长同形。
- **成功：** 一套状态跨过切换点；2b 的 \(V\equiv 0\) 且 regret 无明显 \(\sqrt T\) 斜向上。
- **失败且可修：** 切换后短时跳一下（可接受，标出来）；2b 的 \(V\) 因数值不是精确 0（应用解析梯度，应精确 0）。
- **失败且撤图：** 切换后 regret 爆炸且实现无误 → 与 same-run 叙事冲突，该图不进主文。

### Exp.3 Warm vs restart（主文 Figure 3 或表）

- 游戏：G2，\(\texttt{ell1}=10^{-3}\)（必须真的加倍）。
- 两条学习器其余相同，只改是否 reset。
- 图：\(\beta_t\) 或 \(J_t\)（warm 应冻结）；累计 regret（restart 在重置处出台阶）。
- **成功：** warm 的 \(J\) 符合 \(\lceil\log_2^+(L^{\mathrm{row}}/\ell_1)\rceil\) 量级后不变；restart 更不稳或台阶可见。
- **失败且可修：** 从未加倍 → 再减小 \(\ell_1\) 或加大 \(A,\mu\)。
- **失败且撤图：** restart 看起来一样好。正文本来就不证明 restart 必更差；删图，不改定理表述。

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

**Step 6. Exp.2。** 先 2b（解析最死），再 2a。

**Step 7. Exp.3。** 确认 \(J\ge 1\) 后再对比 restart。

**Step 8. 出图。** `matplotlib`，pdf 矢量；色盲友好、线型可分；图注只写现象与定理编号，不写“优于 Hsieh”。

**Step 9. Go / no-go（见 §7）。** 决定主文放几张图。实现细节进论文 appendix，不进 Introduction。

工时估计（已有公式、从零写代码）：Step 1–4 约半天，Step 5–8 约一天。不要和改主文抢同一周的优先级（主文仍是：一条主定理、对照表、证明进 appendix）。

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
| restart 不更差 | 删 Exp.3 |
| self-play regret 真按 \(\sqrt T\) 涨 | 先查 §3.4 顺序与 \(g^y\) 符号；仍反则停实验、查证明 |
| 切换后崩溃 | 不进主文；论文改回纯理论 |

**原则：** 失败的图比没有图更伤。COLT 直接跳过本文件夹。

---

## 8. 主文怎么引用

放在 Main Theorem 与 limitations 之间。

- Figure 1 = Exp.1（G2 为主，G1 作补充可进 appendix）。
- Figure 2 = Exp.2a + 2b。
- Figure 3 = Exp.3（仅当 doubling 可见且 restart 有反差）。

正文三句话就够：self-play 饱和；同一轨迹切换后不重开；warm 有限次加倍、restart 会断状态。不把实验写成贡献条目。

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
