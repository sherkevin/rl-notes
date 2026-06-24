---
aliases: [Mid-Training, 中训练, 中期训练]
tags:
  - llm
  - mid-training
  - overview
created: 2026-06-23
---

# LLM Mid-Training（中期训练）

> 介于 Pre-Training 和 Post-Training 之间，用 RL 或 curriculum learning 提升模型能力。

---

## 一句话定义

**Mid-Training = 在预训练之后、对齐之前，用强化学习或课程学习提升模型的基础能力**。

与 Post-Training 的区别：
- **Mid-Training**：提升能力（推理、数学、代码）
- **Post-Training**：对齐偏好（听话、安全、有用）

---

## 为什么需要 Mid-Training？

### Pre-Training 的局限

预训练只学"下一个 token"，没有显式优化：
- 推理能力
- 数学能力
- 代码能力
- 长序列理解

### Post-Training 的局限

对齐阶段假设模型已经"会"了，只是让它"听话"：
- 如果模型不会推理，DPO/RLHF 也教不会
- 对齐数据有限，无法覆盖所有能力

### Mid-Training 的定位

```
Pre-Training（学语言）
    ↓
Mid-Training（学能力：推理、数学、代码）← **你在这里**
    ↓
Post-Training（学对齐：听话、安全）
```

---

## 三大方法

### 1. Curriculum Learning（课程学习）

**核心思想**：从简单到难，逐步学习。

$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{easy}} \to \mathcal{D}_{\text{medium}} \to \mathcal{D}_{\text{hard}}$$

**实现**：
- 按难度排序数据
- 动态调整数据混合比例
- 自适应难度

**应用**：
- 数学推理（先学加减，再学乘除）
- 代码生成（先学简单函数，再学复杂算法）

---

### 2. Self-Play（自我对弈）

**核心思想**：模型与自己（或历史版本）对弈，生成高质量训练数据。

**流程**：
1. 模型生成多个回答
2. 用奖励模型或规则打分
3. 选最好的继续训练
4. 重复

**代表**：
- **SPIN**（Self-Play Fine-Tuning）：用 DPO 框架做 self-play
- **Self-Rewarding**：模型自己给自己打分

**优点**：
- 不需要人类标注
- 可以持续改进

---

### 3. RL for Continued Pre-Training（RL 续训）

**核心思想**：用 RL 优化预训练目标，而不是下一个 token prediction。

**奖励信号**：
- 可验证奖励（数学答案正确性）
- 过程奖励（推理步骤质量）
- 外部评分（代码能否运行）

**代表**：
- **DeepSeek-R1**：用 RL 训练推理能力
- **OpenAI o1**：用 RL 训练 chain-of-thought
- **CodeRL**：用 RL 训练代码生成

---

## Mid-Training vs Post-Training

| | Mid-Training | Post-Training |
|---|---|---|
| **目标** | 提升能力 | 对齐偏好 |
| **奖励信号** | 客观（正确性、可执行性） | 主观（人类偏好） |
| **数据** | 自动生成或规则验证 | 人类标注 |
| **阶段** | Pre-Training 之后 | Mid-Training 之后 |
| **代表方法** | SPIN、Curriculum、RL | RLHF、DPO、GRPO |

---

## 2024-2025 前沿

### 1. DeepSeek-R1（推理 RL）

**核心思想**：用 RL 训练 chain-of-thought 推理。

**关键技术**：
- Verifiable reward（答案正确性）
- Process reward（推理步骤质量）
- Long-horizon RL（长序列优化）

**效果**：在数学推理上达到 GPT-4o 水平。

---

### 2. SPIN（Self-Play Fine-Tuning）

**核心思想**：用 DPO 框架做 self-play。

**流程**：
1. 模型 $\pi_0$ 生成回答 $\{y_1, y_2\}$
2. 用规则或 RM 判断哪个好
3. 用 DPO 训练 $\pi_1$
4. $\pi_1$ 再生成，训练 $\pi_2$
5. 重复

**优点**：
- 不需要人类偏好数据
- 可以持续改进

---

### 3. Expert Iteration（专家迭代）

**核心思想**：生成 → 筛选 → 训练 → 生成更好的。

**流程**：
1. 模型生成大量回答
2. 用强规则筛选（如数学答案正确）
3. 用筛选后的数据训练
4. 重复

**应用**：
- AlphaGo（自我对弈 + MCTS）
- 数学推理（生成 + 验证）

---

## 挑战

### 1. 奖励设计

- 推理过程怎么打分？
- 代码质量怎么评估？
- 长序列的奖励稀疏

### 2. 探索 vs 利用

- 模型容易陷入局部最优
- 需要鼓励多样性
- Self-play 可能坍缩

### 3. 计算成本

- RL 训练慢
- 长序列采样贵
- 需要大量计算

---

## 演化位置

```
Pre-Training (2018-2020)
    ↓
SFT (2020-2022)
    ↓
RLHF (2022)
    ↓
Mid-Training (2024-2025) ← **你在这里**
  - Curriculum Learning
  - Self-Play (SPIN)
  - RL for Reasoning (DeepSeek-R1, o1)
    ↓
Post-Training (DPO, GRPO)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]

---

## 参考资源

- 论文：DeepSeek-AI "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- 论文：Chen et al. "Self-Play Fine-Tuning Converts Synthetic Data to Supervised Data" (2024, SPIN)
- 论文：Silver et al. "Mastering the Game of Go without Human Knowledge" (2017, AlphaGo Zero)
- 博客：[Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
