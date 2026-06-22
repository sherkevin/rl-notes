---
tags:
  - module
  - reward-model
  - rlhf
  - generative
created: 2026-06-22
---

# Generative Reward Model

> 先生成评价理由，再输出分数。可解释性强，但推理速度慢。

---

## 组合构成

本模块由以下原子组合而成：
- **生成式评估**：模型输出 critique + score
- **两阶段训练**：先用 LLM 生成数据，再微调小模型

---

## 核心创新

### 思路

传统 RM 是一个黑盒标量输出。Generative RM 让模型**先输出评价理由，再输出分数**：

```
Input: "Prompt: ... Response: ..."
Output: "这个回答的优点是准确、有条理。缺点是缺少具体例子。
         综合评分: 7/10"
```

---

## 算法流程

### 数据生成

1. 用 LLM 生成 (prompt, response, critique, score) 的训练数据
2. critique 是评价理由，score 是最终分数

### 模型训练

1. 微调一个较小模型学习"生成 critique + 输出分数"
2. 用生成的 critique 作为中间监督信号

---

## 优缺点

- ✅ 可解释——能看到模型为什么给这个分
- ✅ critique 本身可以作为过程奖励
- ❌ 推理速度慢（要生成一段文本而不只是一个数）

---

## 代表工作

- **Skywork-Reward** (2024)：generative RM + RLHF
- **Self-Taught Evaluators** (Meta 2024)：让模型自我训练评判能力
- **Generative Reward Models** (2024, arxiv 2410.12832)：证明 generative RM 在某些场景下优于 Bradley-Terry RM

---

## 2025-2026 趋势

- **Generative RM 替代黑盒 RM**：可解释性成为刚需
- critique 可以作为过程奖励，结合 PRM 使用

---

## 演化位置

**在 Reward Model 演化链中的位置**：

```
Pairwise RM (2022)
    ↓
Pointwise RM / PRM / ORM (2023-2024)
    ↓
LLM-as-Judge (2024)
    ↓
Generative RM (2024) ← **你在这里**
    ↓
Verifiable Reward / DPO (2024-2025)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
