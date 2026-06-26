---
tags:
  - llm
  - rl
  - deep-research
  - 2024-2025
  - survey
created: 2026-06-26
---

# LLM 训练全流程中的 RL 技术：深度调研报告

> 基于 24 篇论文的深度调研，覆盖 2024-2026 前沿。106 个 agent 搜索、抓取、对抗验证后的综合结论。

---

## 调研概况

| 指标 | 数据 |
|------|------|
| 搜索角度 | 5 个（数学基础、训练流程、代表系统、前沿方向、实践优化） |
| 抓取来源 | 24 篇论文/博客 |
| 提取声明 | 114 条 |
| 对抗验证 | 25 条（21 确认 ✅，4 否决 ❌） |
| 最终发现 | 10 条高置信度结论 |

---

## 核心发现

### 发现 1：统一策略梯度框架 ⭐⭐⭐

**所有主流 LLM 后训练方法（SFT、[[05-应用领域/Post-Training/RLHF|RLHF]]、RLVR、[[02-模块/Reward-Model/DPO|DPO]]）都是同一个[[01-原子/策略梯度|策略梯度]]框架的特例**，区别仅在于三个组件：梯度系数、数据来源、稳定化机制。

**DPO 的数学本质**：通过 [[01-原子/Bradley-Terry模型|Bradley-Terry]] 偏好模型重参数化奖励函数，将 RLHF 简化为二分类交叉熵损失：

$$r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + C$$

**来源**：arXiv 2407.16216（综述，v4 2026.05）、arXiv 2305.18290（DPO 原始论文）

**置信度**：high（投票 2-1, 3-0）

---

### 发现 2：GRPO 去掉 Critic 模型 ⭐⭐⭐

**[[02-模块/Policy-Based/GRPO|GRPO]] 通过组内相对归一化估计[[01-原子/优势函数|优势函数]]，彻底去掉价值模型（Critic）**，大幅降低训练资源需求。

**优势公式**：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1 \ldots r_G)}{\text{std}(r_1 \ldots r_G)}$$

**实践细节**：每个 rollout batch 只做一次梯度更新。已在 DeepSeek-R1 和 DeepSeek-V3 中验证。

**来源**：arXiv 2402.03300（DeepSeekMath，Section 4.1.1）

**置信度**：high（投票 2-1）

---

### 发现 3：DeepSeek-R1 的 4 阶段流水线 ⭐⭐⭐

**DeepSeek-R1 使用 SFT+RL 交替的 4 阶段流水线**：

```
阶段 1: 冷启动 SFT（少量高质量数据）
    ↓
阶段 2: 推理 RL（GRPO + 规则奖励）← 核心
    ↓
阶段 3: 拒绝采样 SFT（用 RL 模型生成数据再 SFT）
    ↓
阶段 4: 全场景 RL（通用能力对齐）
```

**纯 RL 变体 R1-Zero**（无 SFT，仅 [[02-模块/Policy-Based/GRPO|GRPO]] + 规则奖励）：
- AIME 2024 pass@1 = **71.0%**，cons@64 = **86.7%**
- 匹配 OpenAI o1-0912
- **证明大规模纯 RL 可以产生强推理能力**

**完整流水线性能**：

| 基准 | R1 | V3-SFT | 提升 |
|------|-----|--------|------|
| AIME | 79.8% | 39.2% | +40.6% |
| MATH-500 | 97.3% | — | — |
| Codeforces | 2029 | 1134 | +895 |

**关键洞察**：中间 SFT 阶段实际上会**损害推理能力**（R1-Dev1 在 AIME 上从 77.9% 降到 59.0%）。

**来源**：arXiv 2501.12948（Nature，2025.01）

**置信度**：high（投票 3-0, 2-1, 3-0）

---

### 发现 4：RL 训练产生涌现行为 ⭐⭐

**RL 训练产生涌现的高级推理模式**——自我反思、验证、动态策略调整、自发的"顿悟时刻"——这些行为不是显式编程到奖励或训练数据中的。

**示例**：模型生成 "Wait, wait. That's an aha moment I can flag here."

**注意事项**：
- "涌现"指不在奖励/数据中，但可能是 RL + think-tag 结构的可预测结果
- 示例可能是 cherry-picked
- R1-Zero 有可读性和语言混合问题

**来源**：arXiv 2501.12948 Section 2.2.4

**置信度**：medium（投票 3-0）

---

### 发现 5：RL 产生真正的推理能力提升的条件 ⭐⭐⭐

**RL 只在以下条件下产生真正的推理能力提升（pass@128）**：

1. **预训练留有足够余量**（headroom）
2. **RL 训练数据针对模型的"能力边缘"**——难度适中但不至于无法解决的任务

**实验证据**：

| 任务难度 | pass@128 提升 |
|---------|-------------|
| ID 任务（op=2-10） | 无提升 |
| OOD 边缘任务（op=11-14） | **+42%** |
| OOD 困难任务（op=15-20） | 失败 |

**[[02-模块/Reward-Model/PRM|过程奖励]] vs 结果奖励**：过程级奖励信号减少 reward hacking，提升推理保真度。

**来源**：arXiv 2512.07783（Zhang, Neubig, Yue, 2025.12）、arXiv 2305.20050（OpenAI PRM）

**置信度**：high（投票 3-0, 2-1）

---

### 发现 6：RL 在预训练早期就有效 ⭐⭐

**RL（RLVR with [[02-模块/Policy-Based/GRPO|GRPO]]）从预训练早期检查点（仅 4B token）就有效**，在 1B 模型上将 GSM8K pass@1 从 ~2% 提升到 ~18%。

**数据组成 > 模型规模**：对于更难的问题（MATH 风格），针对性的预训练数据组成比扩大模型规模更有效——添加 10B 数学专用 token 带来的 RL 增益大于同等 token 预算下从 1B 扩展到 4B 参数。

**到 10B token 时，直接 RL 匹配完整的 SFT-then-RL 流水线。**

**来源**：arXiv 2606.04272（Bansal et al., Harvard, 2026.06）

**置信度**：medium（投票 2-1, 3-0）

**注意**：仅在 1B/4B 规模验证，70B+ 外推未证实。

---

### 发现 7：直接 RL 扩展输出分布，SFT 后才 RL 会收窄 ⭐⭐

**直接在基础预训练检查点上应用 RL 会扩展模型输出分布**（pass@1 和 pass@k 都提升）。

而先前报告的"分布收窄效应"（pass@1 上升，pass@k 下降）**仅在 RL 跟随 SFT 之后时出现**——是 SFT 限制了探索，不是 RL 本身。

**机制**："没有预先接触标准答案，模型通过 on-policy 学习探索并发现新的推理路径。"

**注意**：仅在 1B/4B 规模测试；在更难的 MATH 问题上，直接 RL 不如 SFT-then-RL。

**来源**：arXiv 2606.04272 Section 4.1

**置信度**：medium（投票 2-1）

---

### 发现 8：Meta RAM 的 Thinking Mid-Training ⭐⭐⭐

**Meta RAM 团队引入 "thinking mid-training"**——预训练和后训练之间的新中间阶段：

```
步骤 1: 标注模型数据增强（交错思维链）
步骤 2: SFT mid-training（增强数据）
步骤 3: RL mid-training（DrGRPO + LLM-as-Judge 二值奖励）
```

**完整流水线（SFT+RL mid-training → RLVR post-training）**：

| 模型 | 无 mid-training | 完整流水线 | 提升 |
|------|----------------|-----------|------|
| Llama-3-8B | 0.1197 | 0.3837 | **3.2×** |

测试基准：GSM8K、MATH-500、AMC23、Olympiad、GPQA-Diamond

**来源**：Meta FAIR blog + arXiv 2601.21343（2026.01）

**置信度**：high（投票 3-0, 3-0, 3-0）

---

### 发现 9：多解法 Mid-Training 提升 pass@k ⭐⭐

**RL 之前在 mid-training 中使用多样化自生成数据（每题多种解法）可以一致地提升数学推理的 pass@k**。

**在固定预算下**，学习每题多种解法比学习更多题的单解法提升 ~7%（都是 7,408 个实例）。

**RL 训练导致涌现的多解法组合**：56.7% vs 无 RL 的 23.3%（n=16 启发式）。

**注意**：结果仅在 3B 模型上验证；Qwen2.5-7B 未显示一致提升。

**来源**：arXiv 2605.08472（ASU + Google DeepMind, 2026.05）

**置信度**：medium（投票 2-1, 3-0, 2-1, 3-0）

---

### 发现 10：并行平均 RL 和 SFT 梯度 ⭐⭐

**在单个训练步中并行平均 RL 和 SFT 梯度**，在所有预训练检查点上达到最强 pass@32，同时保持非数学通用能力。

**但 pass@1 低于直接 RL 和 SFT 基线。**

**注意**：仅在单示例配方中比较，未与 SFT-Gold（多示例）对比。作者称其为"需要前沿规模验证的早期数据点"。

**来源**：arXiv 2606.04272 Section 5

**置信度**：medium（投票 2-1）

---

## 被否决的声明

| 声明 | 投票 | 原因 |
|------|------|------|
| "DeepSeek-R1 的纯 RL 无需任何监督数据就能产生 CoT" | 0-3 | R1-Zero 仍有冷启动 SFT 阶段的变体；完全无 SFT 的版本可读性差 |
| "RL 方法分两大阵营：传统 RM-based vs 简化版" | 0-3 | 统一框架（发现 1）表明所有方法都是同一框架的特例 |
| "RL 几乎只在后训练阶段使用" | 0-3 | 发现 6、8 证明 RL 在 mid-training 甚至早期预训练检查点就有效 |
| "Mid-training 在固定预算下显著优于纯 RL" | 1-2 | 取决于具体任务和模型规模，不是普遍结论 |

---

## 开放问题

1. **"RL before SFT" 的发现（分布扩展、早期检查点有效性、并行平均）在前沿规模（70B+ 参数、10T+ token）是否成立？** 还是 1B-4B 实验的伪影？

2. **Mid-training vs Post-training RL 的最优调度是什么？** Meta RAM 的 3 步 mid-training 是否普遍优越，还是最优流水线取决于领域、模型规模和基础模型质量？

3. **[[02-模块/Reward-Model/PRM|PRM]] 如何扩展到前沿 RL 训练？** 能否与 [[02-模块/Policy-Based/GRPO|GRPO]] 风格的组内相对优势估计结合，同时获得样本效率和奖励保真度？

4. **纯 RL 的"涌现"推理行为的理论极限是什么？** 自我反思和策略调整是真正的能力提升，还是 think-tag 结构激励的复杂模式匹配？

---

## 关键来源

| 论文 | 年份 | 贡献 |
|------|------|------|
| [arXiv 2407.16216](https://arxiv.org/abs/2407.16216) | 2026.05 | 统一策略梯度框架综述 |
| [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | 2025.01 | DeepSeek-R1 技术报告（Nature） |
| [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) | 2024.02 | DeepSeekMath / GRPO 原始论文 |
| [arXiv 2606.04272](https://arxiv.org/abs/2606.04272) | 2026.06 | RL before SFT（Harvard） |
| [arXiv 2601.21343](https://arxiv.org/abs/2601.21343) | 2026.01 | Meta RAM Thinking Mid-Training |
| [arXiv 2512.07783](https://arxiv.org/abs/2512.07783) | 2025.12 | RL 能力提升的条件（CMU Neubig） |
| [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) | 2023.05 | DPO 原始论文 |
| [arXiv 2305.20050](https://arxiv.org/abs/2305.20050) | 2023.05 | OpenAI 过程奖励模型 |
| [arXiv 2605.08472](https://arxiv.org/abs/2605.08472) | 2026.05 | 多解法 Mid-Training（ASU + DeepMind） |

---

## 对我们知识库的启示

基于调研发现，现有文档需要更新的要点：

1. **02-Mid-Training阶段.md**：
   - 补充 Meta RAM 的 Thinking Mid-Training（发现 8）
   - 补充"RL 从预训练早期就有效"（发现 6）
   - 补充"多解法优于多题目"（发现 9）

2. **03-Post-Training阶段.md**：
   - 补充统一[[01-原子/策略梯度|策略梯度]]框架（发现 1）
   - 更新 [[02-模块/Policy-Based/GRPO|GRPO]] 细节：单次梯度更新（发现 2）
   - 补充 DeepSeek-R1 的 4 阶段流水线细节（发现 3）
   - 补充"SFT 实际上可能损害推理"的反直觉发现（发现 3）
   - 补充"RL 能力边缘"条件（发现 5）

3. **01-Pre-Training阶段.md**：
   - 补充"直接 RL 扩展分布 vs SFT-then-RL 收窄分布"（发现 7）

---

*本报告由 deep-research workflow 生成，106 个 agent 参与，4,776,108 token 消耗。*
