---
aliases: [Post-Training方法对比, LLM对齐方法]
tags:
  - llm
  - post-training
  - comparison
created: 2026-06-23
---

# LLM Post-Training 方法全景

> 从 RLHF 到 DPO 到 GRPO，各种对齐方法的演化脉络和选择指南。

---

## 一句话总览

**Post-training = 让预训练模型"听话"的技术**，核心问题是如何利用人类偏好（或类似信号）优化模型。

---

## 三大流派

### 1. RL-Based（强化学习派）

**代表**：[[05-应用领域/Post-Training/RLHF|RLHF]]、[[02-模块/Policy-Based/GRPO|GRPO]]、[[02-模块/Policy-Based/DAPO|DAPO]]

**流程**：
1. 训练 reward model（或直接用 verifiable reward）
2. 用 RL 算法（PPO 等）优化 policy

**优点**：
- 探索能力强
- 可以处理复杂奖励信号

**缺点**：
- 训练复杂、不稳定
- 计算成本高

---

### 2. Direct Optimization（直接优化派）

**代表**：[[02-模块/Reward-Model/DPO|DPO]]、SimPO、KTO、ORPO、IPO

**流程**：
1. 直接用偏好数据优化 policy
2. 不需要显式训练 reward model

**优点**：
- 简单、稳定
- 计算成本低

**缺点**：
- 探索能力弱
- 对数据质量敏感

---

### 3. Iterative（迭代派）

**代表**：Self-Rewarding、SPIN、Expert Iteration

**流程**：
1. 生成数据 → 训练 → 生成更好的数据 → 再训练...
2. 多轮迭代，自我改进

**优点**：
- 可以持续改进
- 不需要大量人类标注

**缺点**：
- 可能累积误差
- 需要精心设计迭代策略

---

## 方法对比表

| 方法 | 年份 | 需要 RM | 需要 RL | 需要 Value Model | 稳定性 | 适用场景 |
|------|------|---------|---------|------------------|--------|----------|
| [[05-应用领域/Post-Training/RLHF\|RLHF]] | 2022 | ✅ | ✅ (PPO) | ✅ | ❌ | 预算充足、追求最强效果 |
| [[02-模块/Reward-Model/DPO\|DPO]] | 2023 | ❌ | ❌ | ❌ | ✅ | 快速迭代、数据质量高 |
| [[02-模块/Policy-Based/GRPO\|GRPO]] | 2024 | ✅/❌ | ✅ | ❌ | ⚠️ | 内存受限、生成式任务 |
| [[02-模块/Policy-Based/DAPO\|DAPO]] | 2024 | ✅/❌ | ✅ | ✅ | ✅ | 需要解耦优化 |
| SimPO | 2024 | ❌ | ❌ | ❌ | ✅ | 无参考模型 |
| KTO | 2024 | ❌ | ❌ | ❌ | ✅ | 二值反馈（好/坏） |
| ORPO | 2024 | ❌ | ❌ | ❌ | ✅ | SFT + 对齐一步完成 |
| IPO | 2023 | ❌ | ❌ | ❌ | ✅ | DPO 的正则化版本 |

---

## 关键问题：Reward Model 要不要？

### 要 RM（RLHF、GRPO）

**优点**：
- 可以给任意回答打分
- 可以做 process reward（过程奖励）
- 可以集成多个信号

**缺点**：
- 训练 RM 成本高
- RM 可能不准
- Reward hacking 风险

### 不要 RM（DPO、SimPO、KTO）

**优点**：
- 简单直接
- 不需要额外模型
- 更稳定

**缺点**：
- 只能用偏好数据
- 无法做过程奖励
- 探索受限

---

## 关键问题：RL 要不要？

### 要 RL（RLHF、GRPO、DAPO）

**优点**：
- 可以探索新回答
- 可以处理复杂奖励
- 理论上更强

**缺点**：
- 训练不稳定
- 超参数敏感
- 计算成本高

### 不要 RL（DPO、SimPO、KTO、ORPO）

**优点**：
- 训练稳定
- 简单快速
- 易于调试

**缺点**：
- 只能从已有数据学习
- 无法探索
- 可能陷入局部最优

---

## 2024-2025 前沿

### 1. Verifiable Reward

**代表**：[[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]]、数学推理 RL

**核心思想**：用可验证的信号（如数学答案正确性）替代人类偏好。

**优点**：
- 不需要人类标注
- 奖励准确
- 可以大规模扩展

**应用**：
- 数学推理（DeepSeek-R1、OpenAI o1）
- 代码生成（CodeRL）
- 逻辑推理

---

### 2. Process Reward Model（PRM）

**代表**：[[02-模块/Reward-Model/PRM|PRM]]、Math-Shepherd

**核心思想**：不仅给最终答案打分，还给中间步骤打分。

**优点**：
- 更细粒度的监督
- 可以引导推理过程
- 减少 reward hacking

**应用**：
- 数学推理
- 多步推理任务

---

### 3. Self-Rewarding

**代表**：[[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]]、Meta 的 Self-Rewarding Language Models

**核心思想**：让模型自己给自己打分，迭代改进。

**流程**：
1. 生成多个回答
2. 模型自己打分
3. 选最好的继续训练
4. 重复

**优点**：
- 不需要人类标注
- 可以持续改进

**缺点**：
- 可能累积误差
- 需要防止模型崩溃

---

### 4. RL for Reasoning

**代表**：DeepSeek-R1、OpenAI o1、QwQ

**核心思想**：用 RL 训练推理能力（chain-of-thought）。

**关键技术**：
- Verifiable reward（答案正确性）
- Process reward（推理步骤质量）
- Long-horizon RL（长序列优化）

**挑战**：
- 推理序列很长
- 奖励稀疏
- 需要大量计算

---

## 选择指南

### 你有偏好数据 (y_w, y_l) 吗？

- **有** → [[02-模块/Reward-Model/DPO|DPO]]（简单稳定）或 [[05-应用领域/Post-Training/RLHF|RLHF]]（追求最强）
- **没有** → 考虑可验证奖励（数学/代码）或生成偏好数据

### 你有预算训练多个模型吗？

- **有** → RLHF（SFT + RM + RL）
- **没有** → DPO（SFT + DPO，两步）或 ORPO（一步）

### 你的任务需要探索吗？

- **需要**（如创意写作）→ RLHF / GRPO
- **不需要**（如分类、抽取）→ DPO / SimPO

### 你的奖励可验证吗？

- **可验证**（数学、代码）→ GRPO + verifiable reward
- **不可验证**（对话、创意）→ RM + PPO 或 DPO

---

## 演化全景

```
SFT (2017-2020)
    ↓
RLHF (2022, InstructGPT)
    ↓
DPO (2023) ──→ SimPO (2024) ──→ KTO (2024) ──→ ORPO (2024)
    ↓
GRPO (2024) ──→ DAPO (2024)
    ↓
Verifiable Reward + RL (2024-2025, DeepSeek-R1, o1)
    ↓
Process Reward + RL (2025, 推理优化)
    ↓
Self-Rewarding (2025, 自我改进)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]

---

## 参考资源

- 综述：Casper et al. "Open Problems and Fundamental Limitations of RLHF" (2023)
- 论文：Rafailov et al. "Direct Preference Optimization" (2023)
- 论文：Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
- 论文：Meng et al. "SimPO: Simple Preference Optimization" (2024)
- 教程：[Hugging Face RLHF Course](https://huggingface.co/learn/cookbook/en/rlhf)
