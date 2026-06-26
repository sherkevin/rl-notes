---
tags:
  - llm
  - training-pipeline
  - rl-application
  - overview
created: 2026-06-26
---

# LLM 训练全流程与 RL 技术

> 大语言模型训练已从简单的预训练+微调，演变为多阶段、多技术融合的复杂流程。强化学习（RL）在其中扮演着越来越关键的角色。

---

## 一句话总览

**LLM 训练 = Pre-Training（学知识）+ Mid-Training（学能力）+ Post-Training（学对齐）**

三个阶段都不同程度地使用了强化学习技术，从早期的"纯监督学习"逐步演变为"RL 贯穿全流程"。

---

## 完整训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM 训练三阶段流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1: Pre-Training (预训练)                                       │
│  ├── 目标：从海量文本学习语言和世界知识                                │
│  ├── 主要技术：Next-token prediction (自监督)                        │
│  ├── RL 技术：                                                      │
│  │   ├── 课程学习 (Curriculum Learning) - 控制数据难度                │
│  │   └── 数据选择 RL - 学习哪些数据更有价值                           │
│  └── 产出：Base Model (基础模型，有知识但不会对话)                     │
│                                                                     │
│  ↓                                                                   │
│                                                                     │
│  阶段 2: Mid-Training (中期训练)                                     │
│  ├── 目标：学习特定能力（推理、代码、数学）                            │
│  ├── 主要技术：领域数据继续预训练                                     │
│  ├── RL 技术（核心）：                                               │
│  │   ├── 自我对弈 (Self-Play) - 生成高质量训练数据                    │
│  │   ├── 推理 RL - 强化学习训练思维链 (Chain-of-Thought)             │
│  │   └── 过程奖励 (Process Reward) - 奖励推理过程而非仅结果           │
│  └── 产出：Capable Model (有特定能力的模型)                          │
│                                                                     │
│  ↓                                                                   │
│                                                                     │
│  阶段 3: Post-Training (后训练/对齐)                                  │
│  ├── 目标：对齐人类偏好（有用、无害、诚实）                           │
│  ├── 主要技术：SFT + RLHF / DPO                                     │
│  ├── RL 技术（核心）：                                               │
│  │   ├── RLHF (PPO + Reward Model) - 经典人类反馈强化学习            │
│  │   ├── DPO / SimPO - 直接偏好优化，无需奖励模型                    │
│  │   ├── GRPO - 组内相对策略优化                                     │
│  │   └── Constitutional AI - 用规则约束替代人工标注                   │
│  └── 产出：Aligned Model (对齐后的模型，如 GPT-4、Claude)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## RL 在各阶段的角色

| 阶段 | RL 重要性 | 主要 RL 技术 | 解决的问题 |
|------|---------|-------------|-----------|
| **Pre-Training** | ⭐⭐ | 课程学习、数据选择 RL | 提高数据效率，避免低质量数据 |
| **Mid-Training** | ⭐⭐⭐⭐⭐ | 自我对弈、推理 RL、过程奖励 | 学习复杂能力（推理、规划） |
| **Post-Training** | ⭐⭐⭐⭐⭐ | RLHF、DPO、GRPO | 对齐人类偏好，控制模型行为 |

**关键洞察**：
- Pre-Training 阶段 RL 是**辅助角色**，主要用于数据选择
- Mid-Training 阶段 RL 成为**核心驱动力**，是学习推理能力的关键
- Post-Training 阶段 RL 是**标准配置**，几乎所有对齐方法都基于 RL 或受 RL 启发

---

## 三个阶段详解

### 阶段 1: Pre-Training（预训练）

**目标**：从海量文本（数万亿 token）学习语言模式、世界知识、推理能力的基础。

**传统方法**：纯自监督学习，预测下一个 token。

$$\mathcal{L} = -\sum_{i} \log P(x_i | x_{<i})$$

**RL 的应用**：

#### 1.1 课程学习 (Curriculum Learning)

**问题**：数据质量参差不齐，随机采样效率低。

**RL 方法**：用 RL 学习数据采样策略，优先采样"当前模型最需要"的数据。

```
状态：当前模型在各类数据上的表现
动作：选择下一批数据的来源和难度
奖励：模型性能提升速度
```

**实际应用**：
- [[01-原子/课程学习]] 的 LLM 版本
- 从简单文本（儿童读物）逐步过渡到复杂文本（学术论文）

#### 1.2 数据选择 RL

**问题**：不是所有数据都有价值，有些数据甚至是有害的。

**RL 方法**：训练一个"数据选择器"，学习哪些数据对模型最有帮助。

```
状态：数据特征（长度、领域、复杂度）
动作：是否纳入训练集
奖励：使用该数据后模型性能提升
```

**产出**：Base Model，有丰富知识但不会对话，可能输出有害内容。

📖 **详细文档**：[[06-LLM训练全流程/01-Pre-Training阶段]]

---

### 阶段 2: Mid-Training（中期训练）

**目标**：在 Base Model 基础上，学习特定能力（数学推理、代码生成、逻辑规划）。

**为什么需要这个阶段？**
- Pre-Training 学到的是"知识"，不是"能力"
- 推理能力需要大量高质量的推理样本，但人工标注成本极高
- Post-Training 阶段主要学对齐，不适合学复杂能力

**RL 的核心作用**：

#### 2.1 自我对弈 (Self-Play)

**问题**：高质量推理数据稀缺，人工标注不可扩展。

**RL 方法**：让模型自己生成训练数据。

```
流程：
1. 模型生成多个候选回答
2. 用验证器（如数学答案检查器）判断正确性
3. 正确的回答作为正样本，错误的作为负样本
4. 用 DPO/PPO 更新模型
5. 重复，形成"自我改进循环"
```

**代表工作**：
- [[05-应用领域/Mid-Training/SPIN]] (Self-Play Fine-Tuning)
- STaR (Self-Taught Reasoner)

#### 2.2 推理 RL (Reasoning RL)

**问题**：推理过程需要多步骤，仅奖励最终答案不够。

**RL 方法**：用过程奖励模型（PRM）奖励推理的每一步。

```
传统 RLHF：
问题 → 推理过程 → 最终答案 → 奖励（仅看答案对不对）

推理 RL：
问题 → 推理步骤1 → 奖励 → 推理步骤2 → 奖励 → ... → 最终答案 → 奖励
```

**代表工作**：
- DeepSeek-R1：用 RL 训练数学推理，显著超越监督学习
- OpenAI o1：推理能力最强的商业模型

**关键 RL 技术**：
- [[02-模块/Reward-Model/PRM]] (Process Reward Model)
- [[02-模块/Policy-Based/PPO]] 或 [[02-模块/Policy-Based/GRPO]]

#### 2.3 持续预训练 + RL

**问题**：如何在不遗忘预训练知识的情况下学习新能力？

**RL 方法**：在继续预训练的同时，用 RL 约束模型不要偏离太远。

```
损失 = 预训练损失 + λ · KL(当前模型 || Base Model)
```

**产出**：Capable Model，具备特定能力（如数学推理），但仍可能输出不当内容。

📖 **详细文档**：[[06-LLM训练全流程/02-Mid-Training阶段]]

---

### 阶段 3: Post-Training（后训练/对齐）

**目标**：让模型对齐人类偏好——有用（helpful）、无害（harmless）、诚实（honest）。

**为什么这是最关键的阶段？**
- 前两个阶段学到的模型可能输出有害内容、虚假信息、不当建议
- 这是模型面向用户前的最后一道工序
- 决定了模型的"人格"和"价值观"

**标准流程**：

```
Step 1: SFT (Supervised Fine-Tuning)
  └── 用人工标注的高质量对话数据微调模型
  └── 产出：SFT Model，会对话但质量不稳定

Step 2: RLHF / DPO (对齐)
  └── 用人类偏好数据训练模型
  └── 产出：Aligned Model，对齐人类偏好
```

**RL 的核心方法**：

#### 3.1 RLHF (Reinforcement Learning from Human Feedback)

**经典流程**：

```
1. 收集偏好数据：人工标注 (prompt, response_w, response_l)
2. 训练奖励模型：学习人类偏好
3. 用 PPO 优化策略：最大化奖励模型分数
```

**优势**：效果最好，GPT-4、Claude 都用这个方法。

**劣势**：流程复杂，需要训练奖励模型，RL 训练不稳定。

📖 **详细文档**：[[05-应用领域/Post-Training/RLHF]]

#### 3.2 DPO (Direct Preference Optimization)

**创新**：跳过奖励模型，直接用偏好数据优化策略。

**数学直觉**：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**优势**：简单稳定，无需奖励模型。

**劣势**：效果略逊于 RLHF，对数据质量敏感。

📖 **详细文档**：[[05-应用领域/Post-Training/DPO]]

#### 3.3 GRPO (Group Relative Policy Optimization)

**创新**：去掉价值模型，用组内相对奖励替代。

```
对同一个 prompt，生成 N 个回答
计算每个回答相对于组内平均水平的优势
用 PPO 更新策略
```

**优势**：节省显存（无需价值模型），适合大模型。

📖 **详细文档**：[[05-应用领域/Post-Training/GRPO]]

#### 3.4 Constitutional AI (CAI)

**创新**：用规则约束替代人工标注。

```
1. 定义一组规则（"宪法"），如"不要输出有害内容"
2. 让模型自己检查输出是否违反规则
3. 违反的输出作为负样本，用于 DPO 训练
```

**优势**：大幅减少人工标注成本。

**代表**：Claude 系列模型。

**产出**：Aligned Model，可以安全地面向用户。

📖 **详细文档**：[[06-LLM训练全流程/03-Post-Training阶段]]

---

## 真实案例

### GPT-4 训练流程

```
Pre-Training:
  └── 13 万亿 token，课程学习控制数据难度
  └── 产出：GPT-4 Base

Mid-Training:
  └── 代码、数学等领域数据继续预训练
  └── 自我对弈生成推理数据
  └── 产出：GPT-4 Capable

Post-Training:
  └── SFT：人工标注对话数据
  └── RLHF：PPO + 奖励模型
  └── 产出：GPT-4（对齐后）
```

### DeepSeek-R1 训练流程

```
Pre-Training:
  └── 标准预训练
  └── 产出：DeepSeek-V3-Base

Mid-Training（核心创新）:
  └── 推理 RL：用 GRPO + 过程奖励训练数学推理
  └── 自我对弈：生成大量推理数据
  └── 产出：DeepSeek-R1（推理能力极强）

Post-Training:
  └── 标准 RLHF 对齐
  └── 产出：DeepSeek-R1-Chat
```

---

## 技术演进趋势

### 趋势 1: RL 从后向前渗透

```
2022: RL 仅在 Post-Training (RLHF)
2023: RL 扩展到 Mid-Training (STaR, SPIN)
2024: RL 开始用于 Pre-Training (数据选择)
```

### 趋势 2: 从人工标注到自动化

```
2022: 依赖大量人工标注偏好数据
2023: DPO 减少对奖励模型的需求
2024: 自我对弈、Constitutional AI 进一步减少人工
2025: 推理 RL 完全自动化（只需验证器）
```

### 趋势 3: 推理能力成为核心

```
2022: 主要关注对齐（无害、有用）
2023: 开始关注推理（Chain-of-Thought）
2024: 推理能力成为差异化竞争点（DeepSeek-R1, o1）
```

---

## 学习路线

### 如果你关注 LLM 训练，按这个顺序学：

1. **先理解 RL 基础**：
   - [[02-模块/Policy-Based/PPO]] - 最常用的 RL 算法
   - [[01-原子/优势函数]] - 理解优势估计
   - [[02-模块/Reward-Model/]] - 奖励建模的各种方法

2. **再学 Post-Training（最成熟）**：
   - [[05-应用领域/Post-Training/RLHF]] - 经典方法
   - [[05-应用领域/Post-Training/DPO]] - 简化方法
   - [[05-应用领域/Post-Training/GRPO]] - 最新方法

3. **然后学 Mid-Training（最前沿）**：
   - [[05-应用领域/Mid-Training/Mid-Training概览]]
   - [[05-应用领域/Mid-Training/SPIN]] - 自我对弈
   - 推理 RL 相关论文（DeepSeek-R1, o1）

4. **最后了解 Pre-Training（探索中）**：
   - [[01-原子/课程学习]]
   - 数据选择 RL 相关论文

---

## 各阶段文档索引

| 阶段 | 文档 | 内容 |
|------|------|------|
| 总览 | 本文档 | 全流程概览，RL 角色分析 |
| Pre-Training | [[06-LLM训练全流程/01-Pre-Training阶段]] | 课程学习、数据选择 RL |
| Mid-Training | [[06-LLM训练全流程/02-Mid-Training阶段]] | 自我对弈、推理 RL、过程奖励 |
| Post-Training | [[06-LLM训练全流程/03-Post-Training阶段]] | RLHF、DPO、GRPO、Constitutional AI |

---

## 关键 RL 技术速查

| 技术 | 阶段 | 核心思想 | 文档 |
|------|------|---------|------|
| PPO | Post-Training | 近端策略优化，稳定训练 | [[02-模块/Policy-Based/PPO]] |
| DPO | Post-Training | 直接偏好优化，无需奖励模型 | [[05-应用领域/Post-Training/DPO]] |
| GRPO | Post/Mid-Training | 组内相对策略优化，节省显存 | [[05-应用领域/Post-Training/GRPO]] |
| 自我对弈 | Mid-Training | 模型自己生成训练数据 | [[05-应用领域/Mid-Training/SPIN]] |
| PRM | Mid-Training | 过程奖励模型，奖励推理过程 | [[02-模块/Reward-Model/PRM]] |
| 课程学习 | Pre/Mid-Training | 控制数据难度，提高学习效率 | [[01-原子/课程学习]] |
| KL 散度 | 全阶段 | 约束模型不要偏离太远 | [[01-原子/KL散度]] |

---

## 总结

**LLM 训练已经从"预训练+微调"的简单流程，演变为"三阶段+RL 贯穿"的复杂系统。**

- **Pre-Training**：RL 辅助数据选择，提高效率
- **Mid-Training**：RL 驱动能力学习，特别是推理能力
- **Post-Training**：RL 实现对齐，是标准配置

**未来的趋势**：RL 会在更多环节发挥作用，从数据选择到能力学习到对齐，形成完整的 RL-enhanced LLM 训练范式。
