---
aliases: [RLHF, RLHF流程, 人类反馈强化学习]
tags:
  - llm
  - post-training
  - rlhf
  - pipeline
created: 2026-06-23
---

# RLHF（人类反馈强化学习）

> 用人类偏好训练 LLM，让模型输出更符合人类期望。三阶段流程：SFT → RM → RL。

---

## 一句话定义

**RLHF = 监督微调 + 奖励模型 + 强化学习**，三阶段串联，让 LLM 从"会说话"变成"说得好"。

---

## 三阶段流程

### 阶段 1：Supervised Fine-Tuning（SFT）

**目标**：让预训练模型学会基本的对话格式。

**数据**：人类专家写的 (prompt, response) 对。

$$\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in \mathcal{D}_{\text{SFT}}} \log \pi_\theta(y|x)$$

**结果**：得到一个能对话的模型 $\pi_{\text{SFT}}$，但质量参差不齐。

---

### 阶段 2：Reward Model（RM）训练

**目标**：学一个打分模型，能判断回答的好坏。

**数据**：人类标注的偏好对 $(x, y_w, y_l)$，$y_w$ 比 $y_l$ 好。

**模型**：基于 SFT 模型，去掉最后的 LM head，换成一个标量输出头。

$$\mathcal{L}_{\text{RM}} = -\sum_{(x, y_w, y_l)} \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

**结果**：得到奖励模型 $r_\phi(x, y)$，能给任意回答打分。

---

### 阶段 3：RL 优化

**目标**：用 RL 最大化奖励模型的分数，同时不要偏离 SFT 模型太远。

**算法**：通常用 [[02-模块/Policy-Based/PPO|PPO]]。

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[r_\phi(x, y)] - \beta \cdot D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

其中 $\pi_{\text{ref}} = \pi_{\text{SFT}}$。

**结果**：得到最终模型 $\pi_{\text{RLHF}}$。

---

## 为什么需要三阶段？

### 为什么不直接 RL？

- 预训练模型不会对话格式，直接 RL 学不到东西
- SFT 提供了一个好的起点

### 为什么不直接用 RM 做 SFT？

- RM 只告诉你分数，不告诉你怎么生成
- RL 能探索 RM 认为好的区域，找到 SFT 数据里没有的回答

### 为什么要 KL 约束？

- 防止模型 "reward hacking"：找到 RM 的漏洞，拿高分但实际回答很差
- 保持模型的多样性和流畅性

---

## RLHF 的问题

### 1. 流程复杂

需要训练三个模型（SFT、RM、RL），调参复杂。

### 2. RM 质量瓶颈

- RM 本身可能不准
- 人类标注成本高、一致性差
- RM 可能被 reward hacking

### 3. 计算成本高

- RL 阶段需要同时跑 4 个模型（policy、reference、reward、value）
- 内存和计算压力大

### 4. 不稳定

- PPO 本身就不稳定
- 超参数敏感
- 容易崩溃

---

## 替代方案

### [[02-模块/Reward-Model/DPO|DPO]]（Direct Preference Optimization）

**核心思想**：跳过 RM，直接用偏好数据优化 policy。

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**优点**：
- 不需要训练 RM
- 不需要 RL
- 更稳定

**缺点**：
- 对数据质量敏感
- 探索能力弱

### [[02-模块/Policy-Based/GRPO|GRPO]]（Group Relative Policy Optimization）

**核心思想**：用 group sampling 替代 value model。

**优点**：
- 不需要 value model（省内存）
- 适合 LLM（生成多个回答，组内比较）

### [[02-模块/Policy-Based/DAPO|DAPO]]（Decoupled Alignment Preference Optimization）

**核心思想**：解耦 SFT 和 RL 阶段，分别优化。

---

## 实践建议

### 什么时候用 RLHF？

- 有充足的预算训练 RM 和跑 RL
- 需要最强的对齐效果
- 有专业标注团队

### 什么时候用 DPO？

- 预算有限
- 快速迭代
- 数据质量高

### 什么时候用 GRPO/DAPO？

- 内存受限（没有 value model）
- 需要更稳定的训练

---

## 演化位置

```
SFT (2017-2020)
    ↓
RLHF (2022, InstructGPT) ← **你在这里**
    ↓
DPO (2023)：跳过 RM
    ↓
GRPO (2024)：跳过 value model
    ↓
DAPO (2024)：解耦优化
```

详见 [[03-流程/Offline-RL与RLHF演化史]]

---

## 参考资源

- 论文：Ouyang et al. "Training language models to follow instructions with human feedback" (2022, InstructGPT)
- 论文：Christiano et al. "Deep reinforcement learning from human preferences" (2017)
- 论文：Rafailov et al. "Direct Preference Optimization" (2023)
- 教程：[Hugging Face RLHF Course](https://huggingface.co/learn/cookbook/en/rlhf)
