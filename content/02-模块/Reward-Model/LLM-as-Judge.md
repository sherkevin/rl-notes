---
tags:
  - module
  - reward-model
  - rlhf
  - llm-as-judge
created: 2026-06-22
---

# LLM-as-Judge（大语言模型作为评判器）

> 不训 RM，直接用强 LLM 打分或做偏好判断。零训练成本，但推理成本高。

---

## 组合构成

本模块由以下原子组合而成：
- **强 LLM 代理**：GPT-4、Claude 等直接评估
- **Prompt Engineering**：设计评估 prompt

---

## 核心创新

### 思路

用一个强 LLM（GPT-4、Claude）直接给 response 打分或做偏好判断，**不训练任何模型**。

```
System Prompt: "你是一个评估专家。请对以下回答打分 (1-10):"
User: "Prompt: ... Response: ..."
Judge: "Score: 7/10, reasoning: 回答准确但缺少例子..."
```

### 变体

| 变体 | 方式 |
|---|---|
| **Pointwise Judge** | 直接给单个 response 打分 |
| **Pairwise Judge** | 比较两个 response，选一个 |
| **Reference-guided** | 提供参考答案，让 judge 对照评分 |

---

## 算法流程

### 直接评估

1. 设计评估 prompt
2. 用强 LLM 对每个 response 打分
3. 直接使用分数作为奖励信号

### 生成训练数据（更常见）

1. 设计评估 prompt
2. 用强 LLM 批量对比 response 对
3. 生成偏好数据 $\{(x, y_w, y_l)\}$
4. 用偏好数据训一个小 RM

---

## 优缺点

- ✅ 零训练成本，开箱即用
- ✅ 能给出理由（可解释性）
- ❌ **推理成本极高**——每次评分都是一次 LLM 推理
- ❌ 有位置偏好（pairwise 中倾向选第一个/最后一个）
- ❌ 有冗长偏好（倾向给更长的 response 高分）
- ❌ 不可微分，不能直接用在 RL 训练中（通常用来生成训练数据，再训一个小 RM）

---

## 实际怎么用

LLM-as-Judge 主要用于**生成偏好数据**：用 GPT-4 批量对比 response 对，生成偏好数据，再用这些数据训一个小 RM。这比人工标注便宜得多。

### 数据生成流程

```
Step 1: 用 SFT 模型生成多个 response
Step 2: 用 GPT-4 做 pairwise 比较
Step 3: 收集偏好数据 {(x, y_w, y_l)}
Step 4: 用偏好数据训小 RM
Step 5: 用 RM 给 PPO 提供奖励信号
```

---

## 2025-2026 趋势

- **LLM-as-Judge 成为数据生产主力**：用 GPT-4/Claude 批量生成偏好数据，再训小 RM
- **[[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]]**：模型自己给自己打分（见 [[Self-Rewarding]]）

---

## 演化位置

**在 [[01-原子/Reward-Model训练方法|Reward Model]] 演化链中的位置**：

```
Pairwise RM (2022)
    ↓
Pointwise RM / PRM / ORM (2023-2024)
    ↓
LLM-as-Judge (2024) ← **你在这里**
    ↓
Generative RM / Verifiable Reward (2024-2025)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
