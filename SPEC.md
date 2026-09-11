# D005 编码规格（Step 0）

本文件把正文算法钉成编程合同。下一步只实现游戏与度量，不在这里改故事、不加游戏、不写学习器。

对照：`note/sections/setting.tex`，`adaptive.tex`，`main-result.tex`，`appendix_a.tex`。  
`log` = 自然对数。全程 `float64`。

---

## 协议（不可违反）

1. 双方**同时**出招：本轮两个 \(w_t\) 都已决定后，才形成 \(z_t=(x_t,y_t)\) 并给梯度。
2. 玩家 \(i\) 只接收 \(g_t^i\)，不接收对手梯度、\(\Phi\) 值、Hessian。
3. 回合结束后 \(z_t\) 公开，仅用于 \(\chi_t^i\)。\(\chi\) 的分子只用**自己的** \(g_t^i,g_{t-1}^i\)。
4. 算法输入只有 \(\epsilon,\beta_0,\ell_1\)。不读 \(L_F,T,G,V,u\)。
5. Warm：从不重置累加器、clipping、\(\zeta\)、动作。加倍只升 \(\beta\)，且新 \(\beta\) 只用于**下一轮**。

反馈符号（`eq:feedback`）：

\[
g_t^x=\nabla_x\Phi(x_t,y_t),\qquad
g_t^y=-\nabla_y\Phi(x_t,y_t).
\]

Last-gradient（`eq:hint`）：\(h_1=0\)，\(t\ge 2\) 时 \(h_t=g_{t-1}\)。分析用的 ghost \(h_{T+1}=0\) 不进入实现循环。

---

## 默认超参

见 `configs/default.yaml`。初始化（`eq:t006-1`）：

\[
\beta_1=\beta_0+\frac{64\,\ell_1^2}{\beta_0},\qquad
\gamma=\epsilon\beta_1.
\]

\(\gamma\) 此后冻结。之后 \(\beta_t=\beta_0+64\ell_t^2/\beta_0\)（`eq:beta-64`）。

---

## 每人状态

| 符号 | 初值 | 备注 |
|---|---|---|
| \(\epsilon,\beta_0\) | 配置 | 常数 |
| \(\gamma\) | \(\epsilon\beta_1\) | 冻结 |
| \(\ell_t,\beta_t\) | \(\ell_1,\beta_1\) | 仅非降 |
| \(\widehat M,B,\overline V,\alpha\) | \(\gamma,\;4,\;4\gamma^2,\;\epsilon/(\sqrt{4}\log^2 4)\) | `eq:B-V` |
| \(\zeta\) | \(0\) | `eq:zeta` |
| \(w\) | \(0\) | 第 1 轮出原点 |
| \(h\) | \(0\) | |
| \(G_{\mathrm{cum}}\) | \(0\) | \(\sum_{s<t}g_s\) |
| \(z_{\mathrm{prev}},g_{\mathrm{prev}}\) | 空 | \(t=1\) 不算 \(\chi\) |
| \(J\) | \(0\) | 严格加倍次数 |

增量缓存（与求和式等价）：`sum_hat2` \(=\sum\|\widehat\Delta_s\|^2\)，`sum_hat2_over_M2` \(=\sum\|\widehat\Delta_s\|^2/\widehat M_s^2\)。

---

## 径向映射与出招

\(a_t=\sqrt{\overline V_t}\)，\(k_t=\beta_t\alpha_t/a_t\)。

\[
q_t(s)=\frac{a_t}{\beta_t}\log\frac{1+k_t e^{s/a_t}}{1+k_t}.
\]

稳定写法（正文允许）：

```
q = (a / beta) * (logaddexp(0, log(k) + s/a) - log1p(k))
```

\(\Theta_t=h_t+G_{\mathrm{cum}}\)，\(\sigma_t=(\|\Theta_t\|-\zeta_t)_+\)。

\[
w_t=
\begin{cases}
0 & \sigma_t=0,\\
-q_t(\sigma_t)\,\Theta_t/\|\Theta_t\| & \sigma_t>0.
\end{cases}
\]

（`eq:w-cf`。不求根、不积分 \(\mathcal R\)。）

---

## 单轮更新顺序（核心）

玩家在回合开始时已经持有本轮的 \(w_t\)（由上一轮算出；\(t=1\) 为 \(0\)）。

环境：收集双方动作 → 算 \(g_t^x,g_t^y\) → 各玩家 `observe(g_t, z_t)`。`observe` 内部**按下列顺序**，不可重排：

1. 记下本轮出招用的 \(a_t=\sqrt{\overline V}\) 和 \(\widehat M_t\)（旧值）。
2. \(\Delta_t=g_t-h_t\)。若 \(\Delta_t=0\) 则 \(\widehat\Delta_t=0\)，否则  
   \(\widehat\Delta_t=\Delta_t\min\{1,\widehat M_t/\|\Delta_t\|\}\)（`eq:clip`）。
3. \(\lambda_t=2\|\widehat\Delta_t\|^2/a_t\)（分母是本轮出招的 \(a_t\)，不是更新后的 \(\overline V\)）。
4. 更新 clipping（`eq:B-V`）：  
   \(\widehat M\leftarrow\max(\widehat M,\|\Delta_t\|)\)，  
   `sum_hat2_over_M2 += \|\widehat\Delta\|^2 / \widehat M_{\mathrm{old}}^2`，  
   `sum_hat2 += \|\widehat\Delta\|^2`，  
   \(B\leftarrow 4+\) `sum_hat2_over_M2`，  
   \(\overline V\leftarrow 4\widehat M^2+\) `sum_hat2`，  
   \(\alpha\leftarrow\epsilon/(\sqrt{B}\log^2 B)\)。
5. \(\zeta\leftarrow\zeta+\lambda_t\)。
6. 若 \(t\ge 2\)：用公开 \(z_t,z_{t-1}\) 与自己的 \(g_t,g_{t-1}\) 算 \(\chi_t\)（`eq:chi`）。  
   \(\chi>\ell\) 则 \(\ell\leftarrow 2\max(\ell,\chi)\)，\(J\leftarrow J+1\)。  
   然后 \(\beta\leftarrow\beta_0+64\ell^2/\beta_0\)。\(t=1\) 跳过本步（`eq:ell-update`）。
7. \(h\leftarrow g_t\)，\(G_{\mathrm{cum}}\leftarrow G_{\mathrm{cum}}+g_t\)，存 \(z_{\mathrm{prev}},g_{\mathrm{prev}}\)。
8. 用**新**的 \(a,\alpha,\beta,\zeta,\Theta=h+G_{\mathrm{cum}}\) 计算下一轮 \(w\)。

第 6 步的新 \(\beta\) 不得用于本轮已打出的 \(w_t\)。

冻结-\(\beta\) 体检：跳过第 6 步（配置 `adaptive: false`）。其余完全相同。

---

## \(\chi_t\)（`eq:chi`）

\[
\chi_t^i=
\begin{cases}
\|g_t^i-g_{t-1}^i\|\,/\,\|z_t-z_{t-1}\| & z_t\neq z_{t-1},\\
0 & \text{otherwise.}
\end{cases}
\]

分子禁止使用对手梯度。分母是**联合**动作差（公开）。

---

## Cold restart（对照，后做）

与 D005 同一套 \(\chi,\ell,\beta\)。仅当 \(\chi_t>\ell_t\) 时，在更新 \(\ell,\beta\) 之后清零：\(w,h,G_{\mathrm{cum}},\zeta,z_{\mathrm{prev}},g_{\mathrm{prev}}\)；用新 \(\beta_+\) 设 \(\gamma_+\leftarrow\epsilon\beta_+\)，clipping 按初值公式重开。主实验默认不用。

---

## 度量定义（实现时按此，本步不写代码）

- \(\operatorname{Reg}_t^x(u)=\sum_{s\le t}\bigl(\Phi(x_s,y_s)-\Phi(u,y_s)\bigr)\)
- \(\operatorname{Reg}_t^y(v)=\sum_{s\le t}\bigl(\Phi(x_s,v)-\Phi(x_s,y_s)\bigr)\)
- \(\operatorname{LinReg}_t^i(u)=\sum_{s\le t}\langle g_s^i,w_s^i-u\rangle\)
- \(Q_t=\sum_{s<t}\|z_{s+1}-z_s\|^2\)
- \(V_t^x(u)=\sum_{s=2}^t\|\nabla_x\Phi(u,y_s)-\nabla_x\Phi(u,y_{s-1})\|^2\)（只依赖对手 \(y\) 与 comparator）
- \(V_t^y(v)\) 对称，用 \(\nabla_y\Phi(x_s,v)\)
- \(G_t^i=\max_{s\le t}\|g_s^i\|\)
- Restricted gap：测试球半径 \(R=1\)。G1 闭式 \(R(\|A^\top\bar x\|+\|A\bar y\|)\)。G2 用解析式，禁止网格搜索。

默认 comparator：G1/G2 鞍点 \(0\)；G3 的 X 用 \(u^\star=-1/\sqrt{3}\)。

---

## 实现后必须成立的不变量

写学习器时用这些当断言，现在只作合同：

- \(\beta\) 非降；`adaptive: false` 时 \(\beta\) 恒等于 \(\beta_1\)。
- \(t=1\) 不加倍；\(J\) 有限且非降。
- \(\gamma\) 永不改；warm 路径 \(G_{\mathrm{cum}}\) 只累加、不清零。
- \(B\ge 4\)，\(\overline V\ge 4\widehat M^2\)，\(\alpha>0\)，\(a\ge 2\widehat M\)。
- \(q_t(s)\) 对大 \(s\) 有限（`logaddexp`）。
- G3 + \(y\equiv 1\)：\(V_t(u^\star)=0\)，\(1\le G_t^x\le 3\)（度量步再断言）。

---

## 状态

- Step 0：本合同。
- Step 1（已完成）：`src/games.py`，`src/metrics.py`，`scripts/check_games_metrics.py`。
- Step 2（已完成）：冻结 \(\beta\) 闭式核，`src/learner.py`，`scripts/check_frozen_beta.py`。原点是鞍点静止点；体检用偏离鞍点的首步。
- 下一步：打开加倍，不要先写 Exp.1。
