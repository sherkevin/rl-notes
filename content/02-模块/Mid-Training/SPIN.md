---
aliases: [SPIN, Self-Play Fine-Tuning]
tags:
  - llm
  - mid-training
  - self-play
created: 2026-06-23
---

# SPIN（Self-Play Fine-Tuning）

> 用 DPO 框架做 self-play，不需要人类偏好数据。

---

## 一句话定义

**SPIN = 模型自己生成回答 → 自己判断好坏 → 用 DPO 训练 → 重复**。

---

## 动机：偏好数据太贵

[[02-模块/Reward-Model/DPO|DPO]] 需要人类标注的偏好数据 $(y_w, y_l)$：
- 标注成本高
- 一致性差
- 覆盖有限

**SPIN 的解决**：让模型自己生成偏好数据。

---

## 流程

### 第 0 轮

1. 从 $\pi_0$（SFT 模型）生成两个回答 $\{y_1, y_2\}$
2. 用规则或 reward model 判断哪个更好
3. 构造偏好对 $(y_w, y_l)$
4. 用 DPO 训练得到 $\pi_1$

### 第 $t$ 轮

1. 从 $\pi_t$ 生成两个回答 $\{y_1, y_2\}$
2. 用 $\pi_{t-1}$（上一轮模型）作为 reference
3. 判断好坏，构造偏好对
4. 用 DPO 训练得到 $\pi_{t+1}$

$$\mathcal{L}_{\text{SPIN}} = -\log \sigma \left( \beta \log \frac{\pi_{t+1}(y_w|x)}{\pi_t(y_w|x)} - \beta \log \frac{\pi_{t+1}(y_l|x)}{\pi_t(y_l|x)} \right)$$

---

## 与 DPO 的对比

| | DPO | SPIN |
|---|---|---|
| 数据来源 | 人类标注 | 模型生成 |
| Reference model | 固定（SFT 模型） | 动态（上一轮模型） |
| 迭代次数 | 1 次 | 多轮 |
| 持续改进 | ❌ | ✅ |

---

## 关键洞察：Reference Model 动态更新

**DPO**：$\pi_{\text{ref}} = \pi_{\text{SFT}}$（固定）

**SPIN**：$\pi_{\text{ref}} = \pi_t$（每轮更新）

**为什么这样做？**
- 每轮模型都在改进
- 用最新模型作为 reference，约束不会太强也不会太弱
- 类似于 TRPO/PPO 的 trust region

---

## 判断好坏的方法

### 1. Verifiable Reward

**适用**：数学、代码等有明确答案的任务。

$$\text{reward}(y) = \begin{cases} 1 & \text{if answer is correct} \\ 0 & \text{otherwise} \end{cases}$$

### 2. Reward Model

**适用**：对话、创意写作等主观任务。

$$\text{reward}(y) = r_\phi(x, y)$$

### 3. LLM-as-Judge

**适用**：通用任务。

用强模型（如 GPT-4）判断两个回答哪个更好。

---

## 优缺点

### 优点

- ✅ 不需要人类偏好数据
- ✅ 可以持续改进
- ✅ 简单（基于 DPO）

### 缺点

- ❌ 可能累积误差（错误判断会传播）
- ❌ 需要可靠的奖励信号
- ❌ 多轮训练成本高

---

## 实践建议

- **迭代次数**：通常 3-5 轮
- **每轮数据量**：10k-100k 条
- **判断方法**：优先用 verifiable reward，其次 LLM-as-Judge
- **防止坍缩**：加入多样性约束（如温度采样）

---

## 演化位置

```
Self-Play (1990s, TD-Gammon)
    ↓
AlphaGo Zero (2017)：自我对弈 + MCTS
    ↓
Expert Iteration (2018)：生成 + 筛选 + 训练
    ↓
SPIN (2024) ← **你在这里**：DPO + Self-Play
```

详见 [[02-模块/Mid-Training/Mid-Training概览]]

---

## 参考资源

- 论文：Chen et al. "Self-Play Fine-Tuning Converts Synthetic Data to Supervised Data" (2024)
- 代码：[uclaml/SPIN](https://github.com/uclaml/SPIN)
