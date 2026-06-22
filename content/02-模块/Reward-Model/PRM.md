---
tags:
  - module
  - reward-model
  - rlhf
  - process-supervision
created: 2026-06-22
---

# Process Reward Model (PRM)

> 给推理过程的每一步打分。密集奖励信号，在数学推理上显著优于 ORM。

---

## 组合构成

本模块由以下原子组合而成：
- **逐步标注**：每个推理步骤单独打分
- **Pointwise 分类器**：对每一步预测"这一步到最终答案正确的概率"

---

## 核心创新

### 数据格式

```
Prompt: "证明根号2是无理数"
Step 1: "假设根号2是有理数，即 √2 = a/b"  → Reward: 0.9 (好的开始)
Step 2: "那么 2 = a²/b²"                     → Reward: 0.8 (正确推导)
Step 3: "所以 a² = 2b²"                      → Reward: 0.85 (对)
Step 4: "所以 a 和 b 都是偶数"               → Reward: 0.3 (跳步了，需要更多论证)
```

训练数据需要标注员给**每一步**打分（或标对错）。

### 训练目标

PRM 通常训练为 pointwise 分类器——对每一步预测"这一步到最终答案正确的概率"：

$$L(\phi) = -\mathbb{E} \sum_t \left[ y_t \log r_\phi(x, y_{\leq t}) + (1-y_t) \log(1 - r_\phi(x, y_{\leq t})) \right]$$

其中 $y_t \in \{0, 1\}$ 是第 t 步的标注标签。

---

## 算法流程

### 数据收集

1. 用 SFT 模型生成推理过程
2. 人工标注员对每一步打分（或标对错）
3. 收集数据集 $\{(x, y_{\leq t}, y_t)\}$

### 模型训练

1. 初始化 reward model
2. 用 BCE loss 训练
3. 验证：检查 reward model 在 held-out 步骤上的准确率

### 应用到 RL

1. 用 PRM 给 PPO 提供每步的奖励信号
2. PPO 优化策略，最大化累积奖励

---

## 优缺点

- ✅ 密集的奖励信号，每步都有反馈
- ✅ 在数学推理上显著优于 ORM（Let's Verify 论文：PRM 通过率 78% vs ORM 56%）
- ❌ 标注成本极高（每步都要标）
- ❌ 如何定义"一步"在不同任务中不统一

---

## 代表工作

- **Let's Verify Step by Step** (Lightman et al., 2023, OpenAI)：首次提出 PRM
- **Math-Shepherd** (2024)：用 MCTS 自动生成每步的 reward 标注
- **OmegaPRM** (2024)：改进 PRM 数据生成流程

---

## 2024-2025 发展

- PRM 的 reward 可以来自验证器：数学题直接 check 每步是否合法，代码题跑 test case
- 自动生成 PRM 数据：用 MCTS 或其他搜索方法，不需要人工标注

---

## 演化位置

**在 Reward Model 演化链中的位置**：

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
