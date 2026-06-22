---
tags:
  - module
  - reward-model
  - rlhf
  - outcome-supervision
created: 2026-06-22
---

# Outcome [[01-原子/Reward-Model训练方法|Reward Model]] (ORM)

> 只看最终结果打分。标注简单，但有 credit assignment 问题。

---

## 组合构成

本模块由以下原子组合而成：
- **结果标注**：只看最终答案对错
- **Pointwise 分类器**：预测最终答案正确的概率

---

## 核心创新

### 数据格式

```
Prompt: "证明根号2是无理数"
Response: [10 行推理过程] → 结论: "所以根号2是无理数"
Reward: 1.0 (正确) 或 0.0 (错误)
```

### 训练目标

$$L(\phi) = -\mathbb{E} \left[ y \log r_\phi(x, y) + (1-y) \log(1 - r_\phi(x, y)) \right]$$

其中 $y \in \{0, 1\}$ 是最终答案的标签。

---

## 算法流程

### 数据收集

1. 用 SFT 模型生成完整推理过程
2. 人工标注员检查最终答案对错
3. 收集数据集 $\{(x, y, y)\}$

### 模型训练

1. 初始化 reward model
2. 用 BCE loss 训练
3. 验证：检查 reward model 在 held-out 数据上的准确率

### 应用到 RL

1. 用 ORM 给 [[02-模块/Policy-Based/PPO|PPO]] 提供最终奖励信号
2. [[02-模块/Policy-Based/PPO|PPO]] 优化策略，最大化最终奖励

---

## 优缺点

- ✅ 标注简单（对/错二分类）
- ❌ **credit assignment 问题**：10 步推理，哪一步做对了？哪一步做错了？ORM 不知道
- ❌ 奖励稀疏：错了就是 0，没有中间信号

---

## 与 PRM 的对比

| 维度 | ORM | PRM |
|---|---|---|
| 奖励粒度 | 最终结果 | 每一步 |
| 标注成本 | 低 | 高 |
| 奖励稀疏 | 是 | 否 |
| credit assignment | 难 | 易 |
| 数学推理效果 | 56% 通过率 | 78% 通过率 |

---

## 演化位置

**在 [[01-原子/Reward-Model训练方法|Reward Model]] 演化链中的位置**：

```
Pairwise RM (2022)
    ↓
Pointwise RM (2023)
    ↓
PRM / ORM (2024) ← **你在这里**
    ↓
LLM-as-Judge / Generative RM (2024)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
