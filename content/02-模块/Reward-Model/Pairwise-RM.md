---
tags:
  - module
  - reward-model
  - rlhf
  - pairwise
created: 2026-06-22
---

# Pairwise [[01-原子/Reward-Model训练方法|Reward Model]]

> 通过成对比较训练 reward model。InstructGPT (2022) 的做法，目前最主流的方式。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/Bradley-Terry模型]]：偏好建模的理论基础
- **人工标注偏好对**：数据格式和标注流程

---

## 核心创新

### 数据格式

人工标注员看到一个 prompt 和**两个** response，选一个更好的：

```
Prompt: "解释什么是黑洞"
Response A: "黑洞是时空中的一个区域..."  ← 标注员选这个
Response B: "黑洞就是一个很黑很黑的洞..."
```

### 训练目标

直接应用 [[01-原子/Bradley-Terry模型]]：

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

Loss 函数：

$$L(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**直觉**：让赢的 response 得分尽量高于输的，差距越大越好。

---

## 算法流程

### 数据收集

1. 用 SFT 模型对每个 prompt 生成多个 response（通常 2-4 个）
2. 人工标注员对 response 对进行成对比较，选出更好的
3. 收集偏好数据集 $\{(x, y_w, y_l)\}$，通常 10k-100k 条

### 模型训练

1. 初始化 reward model（通常从 SFT 模型开始）
2. 用 [[01-原子/Bradley-Terry模型|Bradley-Terry]] loss 训练
3. 验证：检查 reward model 在 held-out 偏好对上的准确率

### 应用到 RL

1. 用训练好的 reward model 给 [[02-模块/Policy-Based/PPO|PPO]] 提供奖励信号
2. [[02-模块/Policy-Based/PPO|PPO]] 优化策略，最大化 reward model 的打分

---

## 优缺点

- ✅ 人类标注简单直观（二选一比打分容易）
- ✅ [[01-原子/Bradley-Terry模型|Bradley-Terry]] 模型理论成熟
- ✅ 能捕捉相对偏好（"A 比 B 好"比"给 A 打 7 分"更容易判断）
- ❌ 只学到**序（ranking），没学到绝对值**——两个 response 差 0.1 分和差 10 分，模型不知道
- ❌ 标注成本高，每个偏好对都需要人工比较

---

## 代表工作

- **InstructGPT** (Ouyang et al., 2022)：首个将 Pairwise RM 应用于 LLM 对齐的大规模工作
- **Anthropic HH-RLHF**：17 万条对话偏好数据
- **UltraFeedback**：用 GPT-4 标注的通用偏好数据

---

## 演化位置

**在 [[01-原子/Reward-Model训练方法|Reward Model]] 演化链中的位置**：

```
人工设计 Reward
    ↓
Pairwise RM (2022) ← **你在这里**
    ↓
Pointwise RM / PRM / ORM (2023-2024)
    ↓
LLM-as-Judge / Generative RM (2024)
    ↓
Verifiable Reward / DPO (2024-2025)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
