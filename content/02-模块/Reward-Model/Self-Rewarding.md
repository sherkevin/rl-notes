---
tags:
  - module
  - reward-model
  - rlhf
  - self-improvement
created: 2026-06-22
---

# Self-Rewarding（自我奖励） Model

> 模型自己给自己打分。减少对人工标注的依赖，但容易"自嗨"。

---

## 组合构成

本模块由以下原子组合而成：
- **自我评估**：LLM 评估自己的输出质量
- **迭代改进**：每轮模型变强 → 偏好数据质量更高

---

## 核心创新

### 思路

让 LLM 自己评估自己的输出质量：

```
Prompt: "评价以下回答的质量 (1-5 分):"
Response: "..."
Self-score: "4/5, because..."
```

---

## 算法流程

### 迭代过程

1. 模型生成多个 response
2. 模型自己评估哪个更好（[[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] on itself）
3. 用这些偏好数据微调模型
4. 回到步骤 1，用更强的模型生成更好的偏好数据

### 训练循环

```
Round 0: SFT 模型 π_0
    ↓
Round 1: π_0 生成 response 对 → π_0 自评 → 偏好数据 → 微调 → π_1
    ↓
Round 2: π_1 生成 response 对 → π_1 自评 → 偏好数据 → 微调 → π_2
    ↓
...
```

---

## 优缺点

- ✅ 减少对人工标注的依赖
- ✅ 可以无限迭代（每轮模型变强 → 偏好数据质量更高）
- ❌ 容易"自嗨"——模型倾向给自己生成的内容高分
- ❌ 需要外部校准防止 reward hacking

---

## 代表工作

- **Self-Rewarding Language Models** (Meta 2024)：模型同时作为策略和 reward model，迭代自我改进
- **Constitutional AI** (Anthropic)：模型根据一组原则自我评估

---

## 演化位置

**在 [[01-原子/Reward-Model训练方法|Reward Model]] 演化链中的位置**：

```
LLM-as-Judge (2024)
    ↓
Self-Rewarding (2024) ← **你在这里**
    ↓
Generative RM / Verifiable Reward (2024-2025)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
