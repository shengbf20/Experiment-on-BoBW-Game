# 实验代码

本目录只保留当前有效的 D005 实验代码、紧凑结果和论文图。旧 `PLAN.md` 与
`SPEC.md` 已删除，避免其中已经撤销的实验与当前安排冲突。

当前信息来源只有：

- `experiment_pruning_plan.md`：现有实验的保留、删除与精简范围；
- `separation_experiment_plan.md`：后续 Separation Example 的实验步骤与产出；
- `../note/v1/appendix_a.tex`：Separation Example 的精确构造与渐近证明；
- `src/separation.py`：Separation Example 的博弈、open-loop opponent 递归和
  comparator 定义（独立于 G1/G2）；
- `src/` 与 `scripts/check_*.py`：当前 D005 实现及可执行不变量检查。

Separation Example 的 Part A 入口是 `scripts/exp_separation.py`，不会改写
Exp.1–3 或 L_F sweep 的结果。旧 stationary G3 不能作为该实验的替代实现。

Part A 已采用改进 construction 收口：取非零平移 comparator
(u^\star=1/4)，并在严格证明允许的范围内取 (\eta=\delta_\star)。冻结回放、精确
(V_T(u^\star)=1)、有界 (G_T) 与线性 tail 机制均通过；在
(10^4\le T\le10^5) 上，原始 (E_T) 的 log-log slope 为 (0.827)，已直接显示有限样本
近线性增长。唯一正式候选图展示原始 (V_T,G_T,E_T) 和 (E_T/T)，不使用 tail proxy
替代 certificate。各 NPZ 已保存新版冻结 `y`，汇总同时记录其
`frozen_y_sha256`。Part B 必须直接回放并核对这些序列，不得重新按 baseline 轨迹构造
opponent。

## 当前实验结构

正文：

1. **Self-play G2**：individual regret 与 movement saturation，以及 restricted gap。
2. **Same-run switch G2**：同一 learner 不做 regime detection、不 reset 地从 self-play
   切换到 arbitrary-opponent regime。

附录：

3. **Unknown-\(L_F\) adaptation + warm-vs-restart**：保留
   \(A=cI, c\in\{0.5,1,2,4\}\) 的代表性 scale sweep；\(c=4\) 另有
   \(T=10^5\) 长跑。warm-vs-restart 只保留 constant-opponent 设置。
4. **Gaussian self-play robustness**：只保留 G1、seed 1 这一非退化实例。

Hsieh 对照、G1 identity、G3 stationary、\(V_T\) sweep、独立 multi-horizon、
以及重复的 Gaussian / warm-restart 设置已移除。Hsieh 对照所得的历史结论见
`../note/experiment_log.md`。

## 仓库边界与环境

- 外层论文仓库忽略整个 `experiment/`；本目录是独立 Git 仓库。
- `results/*` 只在本地保存，`figures/*.pdf` 与 `figures/*.png` 由内层仓库跟踪。
- Python 3.12+，CPU；依赖见 `requirements.txt`。
- 默认设置见 `configs/default.yaml`。投稿实例鞍点为
  \((a,b)=(0.4e_1,0.4e_2)\)，初始化固定为 \(w_1=0\)。

结果写入统一使用 `src/io_results.dump_compact`：JSON 只保存 meta、summary 与
\(T\)，曲线保存为 NPZ。

## 正确性检查

在 `experiment/` 下运行：

```text
python scripts/check_games_metrics.py
python scripts/check_frozen_beta.py
python scripts/check_doubling.py
python scripts/check_closedform.py
python scripts/check_restart.py
python scripts/check_io_results.py
python scripts/check_separation.py
```

## 复现实验

以下命令会运行 learner 并覆盖对应本地结果：

```text
python scripts/exp1_selfplay.py
python scripts/exp2_bobw.py
python scripts/exp3_restart.py
python scripts/exp_lf_sweep.py
python scripts/exp_lf_sweep.py --T 100000 --c 4
python scripts/exp_separation.py
```

以下命令只读取已有结果并作图，不重跑 learner：

```text
python scripts/plot_exp1.py
python scripts/plot_exp2.py
python scripts/plot_exp3.py
python scripts/exp_lf_sweep.py --assemble
python scripts/plot_lf_sweep.py
python scripts/exp_separation.py --assemble
python scripts/plot_separation.py
```

主要产物：

| 实验 | 结果 | 图 |
|---|---|---|
| Self-play G2 | `results/exp1_G2_identity.*` | `figures/exp1_selfplay.*` |
| Gaussian robustness | `results/exp1_G1_gaussian_seed1.*` | `figures/exp1_G1_gaussian.*` |
| Same-run switch G2 | `results/exp2_G2_switch.*` | `figures/exp2_bobw.*` |
| Warm vs restart | `results/exp3_const_{warm,restart}.*` | `figures/exp3_restart.*` |
| Unknown-\(L_F\) | `results/exp_lf_c*.*`、`exp_lf_sweep.json` | `figures/exp_lf_sweep.*` |
| Separation Part A | `results/exp_sep_T*.*`、`exp_sep_partA.json` | `figures/exp_sep_partA.*` |
