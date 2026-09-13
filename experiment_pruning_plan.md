# 实验精简方案（不含 Separation Example）

## 1. 保留在正文的实验

| 实验 | 主要目的 | 是否完整保留 | 调整建议 |
|---|---|---:|---|
| **Self-play G2** | 展示 self-play 下 individual regret 饱和、movement saturation，以及 restricted gap 与 \(O(1/T)\) 行为一致 | **基本完整保留** | 精简图和文字，只保留最能支撑主结论的指标；避免重复展示多个等价的诊断量 |
| **Same-run switch G2** | 展示同一 learner 在不做 regime detection、不 reset 的情况下，从 self-play 进入 arbitrary-opponent regime 后仍保持稳定 | **完整保留** | 强调 “same run / no reset / no regime detection” 这一核心卖点；删去与主结论无关的次要轨迹量 |

---

## 2. 保留在附录的实验

| 实验 | 主要目的 | 是否完整保留 | 需要做的调整 |
|---|---|---:|---|
| **Unknown-\(L_F\) adaptation** | 展示在未知 smoothness scale 时，算法可以自适应调整并正常运行 | **不完整保留** | 保留最有代表性的 \(A=cI\) scale sweep；删去重复或信息增量很小的尺度点；不要声称实验验证了 “multiple doublings / no epoch-count tax”，因为当前实验主要只展示了 scale adaptation |
| **Warm vs restart** | 展示 warm adaptation 相比 cold restart 能更好地保留 learner state，避免重新爬升 | **不完整保留** | 与 Unknown-\(L_F\) 合并成一个机制实验组；只保留最能体现 warm 与 restart 差异的设置和指标；不必单独扩展成大规模实验 |
| **Gaussian self-play robustness check** | 排除 \(A=I\) 等特殊结构，说明 self-play 现象不是退化 toy case | **不完整保留** | 只保留一个代表性的 Gaussian game；优先保留能避免 identity case 中 regret 恒为 0 等退化现象的设置；无需同时保留多个 Gaussian G1/G2 版本 |

建议将前两项合并为一个 appendix 小节，例如：

> **Adaptation to Unknown Smoothness and the Effect of Warm Rescaling**

Gaussian robustness 单独作为一个很短的 robustness check 即可。

---

## 3. 可以删除的实验

以下实验建议从最终论文中删除：

- **G1 identity self-play**
- **现有 G3 stationary \(V_T=0\) 实验**
- **\(V>0\) amplitude sweep**
- **G3 nonstationary low-\(V\) 实验**
- **单独的 multi-horizon terminals 实验**
  - 如有需要，可将少量 multi-horizon terminal values 并入正文的 Self-play G2，而不再作为独立实验
- **所有 Hsieh baseline 对照实验**
- **Hsieh self-play sanity check**
- **Hsieh switch replay**
- 其他仅用于开发阶段、implementation sanity check，但不直接支持论文核心结论的图表

---

## 4. 精简后的实验结构

最终实验部分建议只保留以下结构：

### 正文
1. **Self-play G2**
2. **Same-run switch G2**

### 附录
3. **Unknown-\(L_F\) adaptation + warm-vs-restart**
4. **Gaussian self-play robustness check**

这样可使现有实验部分从“覆盖很多情形”转为更清晰的主线：

\[
\text{self-play behavior}
\;\rightarrow\;
\text{same-run BoBW behavior}
\;\rightarrow\;
\text{unknown-smoothness mechanism / robustness}.
\]

Separation Example 实验作为后续新增内容单独安排，不计入本清单。
