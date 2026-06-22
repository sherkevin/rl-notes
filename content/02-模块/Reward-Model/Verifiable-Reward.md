---
tags:
  - module
  - reward-model
  - rlhf
  - verifiable
created: 2026-06-22
---

# Verifiable Reward

> 规则验证，不需要训练。适用于有确定性答案的任务（数学、代码）。

---

## 组合构成

本模块由以下原子组合而成：
- **规则验证**：检查答案是否正确
- **自动化评估**：无需人工或模型

---

## 核心创新

### 思路

对于有**确定性答案**的任务，直接用规则验证：

| 任务 | 验证方式 |
|---|---|
| 数学 | 检查最终答案是否正确（exact match 或 sympy 验证）|
| 代码 | 跑 test cases，看 pass rate |
| 格式化输出 | 正则匹配格式是否正确 |
| 事实性 | 对照知识库 check |

### 在 RL 中的使用

这就是 DeepSeek-R1 和 GRPO 系方法用的奖励：

```python
def reward_fn(prompt, response):
    answer = extract_answer(response)
    ground_truth = get_ground_truth(prompt)
    return 1.0 if answer == ground_truth else 0.0
```

---

## 算法流程

### 直接应用

1. 定义验证规则（如 exact match、test cases）
2. 对每个 response，自动检查是否正确
3. 直接用 0/1 作为奖励信号

### 结合 GRPO

```
Step 1: 对每个 prompt，生成多个 response
Step 2: 用 verifiable reward 给每个 response 打分 (0 或 1)
Step 3: 用 GRPO 的组内标准化计算优势
Step 4: 更新策略
```

---

## 优缺点

- ✅ 完全不需要训练 reward model
- ✅ 奖励信号完美（对就是对，错就是错）
- ❌ 只适用于有确定性答案的任务
- ❌ 无法评估开放式回答（写作、对话、创意）

---

## 代表工作

- **DeepSeek-R1**：数学/代码用 verifiable reward
- **OpenAI o1/o3**：推理链 + verifiable reward
- **Search-R1** (2025)：用检索准确率作为 verifiable reward

---

## 2025-2026 趋势

越来越多的工作用 **verifiable reward + RL** 来训练推理能力，绕过 reward model 训练：
- DeepSeek-R1：数学/代码用 verifiable reward
- OpenAI o1/o3：推理链 + verifiable reward
- **Search-R1** (2025)：用检索准确率作为 verifiable reward

---

## 演化位置

**在 Reward Model 演化链中的位置**：

```
Pairwise RM (2022)
    ↓
Pointwise RM / PRM / ORM (2023-2024)
    ↓
LLM-as-Judge / Generative RM (2024)
    ↓
Verifiable Reward (2024-2025) ← **你在这里**
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
