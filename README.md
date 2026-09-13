# 实验代码

本目录实现正文算法（D005 / `thm:t006`）和 Hsieh et al. (2021) Euclidean OptDA 对照。合同：`SPEC.md`。步骤：`PLAN.md` §6。评价：`reflection.txt`。

## 仓库边界

- **外层论文仓库**（`Research Loop - Game`）的 `.gitignore` 忽略整个 `experiment/`。主仓不跟踪实验代码、结果或图。
- **内层** `experiment/` 可以单独用 git：跟踪源码与 `figures/*.pdf`、`figures/*.png`；`results/*` 被内层 `.gitignore` 排除，只在本地保留 compact json + npz。

复现时在本目录运行下面的命令，不要从外层仓库根目录跑。

## 环境

Python 3.12+，CPU。依赖见 `requirements.txt`：`numpy`、`matplotlib>=3.8`、`pyyaml`。不要对已有的全局 numpy 2.x 再执行一次裸的 `pip install numpy`。

默认超参在 `configs/default.yaml`：\(\epsilon=1\)，\(\beta_0=1\)，\(\ell_1=10^{-3}\)，\(T=2\times 10^4\)，`hsieh_tau: 1.0`。Hsieh 的 \(\tau\) 是原文可自由选取的常数，**禁止**对着 D005 曲线搜索。

投稿实例鞍点 \((a,b)=(0.4 e_1, 0.4 e_2)\)。脚本禁止在学习器构造之后改 `action`（\(w_1=0\)）。

## 结果格式

新 run **一律**走 `src/io_results.dump_compact`：json 只有 `{meta, summary, T}`，**payload 禁止带 `hist`**（hist 只当第三个参数）。\(T=2\times 10^4\) 主图保持密采样 npz；长跑（G1 \(T=2\times 10^5\)，\(L_F\) 的 `_long`）写 stride-10 下采样 npz，并传 `keys=`。若本地还留着旧的全程 json / 未下采样长跑 npz，可运行一次

```text
python scripts/compact_results.py
```

它会剥掉 `hist`、把平铺长跑 json 收成 `{meta, summary, T}`、对 `*_long.npz` 做 stride-10；已有的 G1 长跑下采样曲线不会被全程 hist 覆盖。

## 体检（不画主图）

在 `experiment/` 下按顺序：

```text
python scripts/check_games_metrics.py
python scripts/check_frozen_beta.py
python scripts/check_doubling.py
python scripts/check_closedform.py
python scripts/check_restart.py
python scripts/check_hsieh.py
python scripts/check_io_results.py
```

PowerShell 不要用 `&&`；一条失败就停：

```text
python scripts/check_games_metrics.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 主实验与作图

每条 `exp*.py` 只写结果；对应 `plot_*.py` 读盘作图，不重跑学习器。

| 步骤 | 命令 | 产物 |
|---|---|---|
| Exp.1 self-play | `python scripts/exp1_selfplay.py` | `results/exp1_*.json/.npz` |
| Exp.1 G1 长跑 \(T=2\times10^5\) | `python scripts/exp1_g1_long.py` | `exp1_G1_identity_long.*` |
| Figure 1 | `python scripts/plot_exp1.py --appendix` | `figures/exp1_selfplay.*`，Gaussian 附录图 |
| Exp.2 switch + G3 | `python scripts/exp2_bobw.py` | `exp2_G2_switch.*`，`exp2_G3_const.*` |
| Figure 2 | `python scripts/plot_exp2.py` | `figures/exp2_bobw.*`（右图若有 Hsieh 则叠加） |
| Exp.3 warm / restart | `python scripts/exp3_restart.py --only const` | `exp3_const_warm.*`，`exp3_const_restart.*` |
| Figure 3 | `python scripts/plot_exp3.py` | `figures/exp3_restart.*` |
| Hsieh 对照 | `python scripts/exp_hsieh.py` | `hsieh_G2_selfplay.*`，`hsieh_G3_const.*`，`hsieh_G2_switch.*` |
| Hsieh 图 | `python scripts/plot_hsieh.py` | `hsieh_selfplay.*`，`hsieh_G3.*`，`hsieh_switch.*` |
| Step 15 horizon 表 | `python scripts/exp_horizon.py` | `results/horizon_table.json`；附录 `tab:exp-horizon` |
| Step 16 \(L_F\) sweep | `python scripts/exp_lf_sweep.py` | `exp_lf_c*.json/.npz`，`exp_lf_sweep.json`（表是 \(T=2\times10^4\)） |
| Step 16 \(c=4,8\) 长跑 | `python scripts/exp_lf_sweep.py --T 100000 --c 4 8` | `exp_lf_c4_long.*`，`exp_lf_c8_long.*`（附录 \(80.76\) / \(\|x_T-a\|=0.033\)） |
| Step 16 图 | `python scripts/plot_lf_sweep.py` | `figures/exp_lf_sweep.*`；附录 `tab:exp-lf`（有 `_long` 则叠到 \(T=10^5\)） |
| Step 17 \(V_T\) | `python scripts/exp_vt_sweep.py` | `exp_vt17a_*.json/.npz`，`exp_vt17b_*.json/.npz` |
| Step 17 图 | `python scripts/plot_vt_sweep.py` | `figures/exp_vt_17a.*`，`exp_vt_17b.*`；附录 `app:exp-vt` |

可选：`python scripts/diag_g1_Q.py` 从 npz 打印 \(Q_t\) 诊断（尊重 stride / `t`，不要把 `len(Q)` 当 \(T\)）。**不要**用它核 `tab:exp-horizon`；核对应走 `python scripts/exp_horizon.py`。`exp3_restart.py` 默认还会跑 self-play ablation；主图只需要 `--only const`。

G3 上 D005 与 Hsieh **双方都饱和**（约 80 vs 1.2）。主图纵轴按 D005 缩放，Hsieh 贴在零轴；相对高度读终端数字，不要读成 realized \(\Omega(\sqrt{T})\)，也不要把 \(1.2<80\) 读成算法排名。Hsieh G2 self-play 图只作实现体检，不是 BoBW 对照。`check_hsieh.py` 的 \(T=4000\) 是松的 smoke，不是独立 rate 证明。

## 一键顺序（Step 11–18）

已完成的 11–18 对应上面的 Exp.2 / Exp.3 / Hsieh / README / horizon 表 / \(L_F\) sweep / \(V_T\)，以及 `note/sections/experiments.tex` 收口。从本目录按检查 → 实验 → 作图即可复现当前主图与附录表。

```text
python scripts/check_games_metrics.py
python scripts/check_frozen_beta.py
python scripts/check_doubling.py
python scripts/check_closedform.py
python scripts/check_restart.py
python scripts/check_hsieh.py
python scripts/check_io_results.py
python scripts/exp1_selfplay.py
python scripts/exp1_g1_long.py
python scripts/plot_exp1.py --appendix
python scripts/exp2_bobw.py
python scripts/exp3_restart.py --only const
python scripts/exp_hsieh.py
python scripts/plot_exp2.py
python scripts/plot_exp3.py
python scripts/plot_hsieh.py
python scripts/exp_horizon.py
python scripts/exp_lf_sweep.py
python scripts/exp_lf_sweep.py --T 100000 --c 4 8
python scripts/plot_lf_sweep.py
python scripts/exp_vt_sweep.py
python scripts/plot_vt_sweep.py
```

Step 18 已改 `note/sections/experiments.tex` 与 `appendix_experiments.tex`。`PLAN.md` 完成判据表已勾满。G3 上 D005 与 Hsieh 双方饱和，不得写成 realized \(\Omega(\sqrt{T})\) 优势。
