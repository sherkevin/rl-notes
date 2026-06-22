---
tags:
  - module
  - reward-model
  - rlhf
  - pointwise
created: 2026-06-22
---

# Pointwise [[01-原子/Reward-Model训练方法|Reward Model]]

> 直接回归标量分数。能学到绝对值，但人类打分不一致。

---

## 组合构成

本模块由以下原子组合而成：
- **直接回归**：MSE loss 训练 reward model
- **人工标注分数**：每个 response 单独打分

---

## 核心创新

### 数据格式

每个 (prompt, response) 对直接给一个标量分数：

```
Prompt: "解释什么是黑洞"
Response: "黑洞是时空中的一个区域..."
Score: 7.5 / 10
```

### 训练目标

直接回归：

$$L(\phi) = \mathbb{E}_{(x, y, s)} \left[ (r_\phi(x, y) - s)^2 \right]$$

**直觉**：让 reward model 的打分尽量接近人工标注的分数。

---

## 算法流程

### 数据收集

1. 用 SFT 模型对每个 prompt 生成 response
2. 人工标注员对每个 response 单独打分（如 1-10 分）
3. 收集数据集 $\{(x, y, s)\}$

### 模型训练

1. 初始化 reward model
2. 用 MSE loss 训练
3. 验证：检查 reward model 在 held-out 数据上的 MSE

---

## 优缺点

- ✅ 能学到绝对分数（知道"好"和"非常好"的差距）
- ❌ 人类打分**极其不一致**——不同标注员对同一个 response 可能给 3 分和 8 分
- ❌ 标注员很难校准自己的标准

---

## 实际怎么用

实践中很少纯用 pointwise，通常是 **pairwise 为主 + pointwise 为辅**：先用 pairwise 训 ranking，再用少量 pointwise 数据校准绝对值。

---

## 演化位置

**在 [[01-原子/Reward-Model训练方法|Reward Model]] 演化链中的位置**：

```
Pairwise RM (2022)
    ↓
Pointwise RM (2023) ← **你在这里**
    ↓
PRM / ORM (2024)
    ↓
LLM-as-Judge / Generative RM (2024)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
