---
tags:
  - module
  - reward-model
  - rlhf
  - llm-alignment
created: 2026-06-22
---

# DPO (Direct Preference Optimization)

> 直接从偏好数据优化策略，无需训练 reward model。策略模型本身就是隐式 reward model。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/KL散度]]：DPO 的隐式 reward 基于 KL 推导
- [[01-原子/Bradley-Terry模型]]：偏好建模的理论基础
- **隐式 reward 技巧**：用策略概率比替代显式 reward model

---

## 核心创新

### 传统 RLHF 的痛点

```
Step 1: SFT          → 用人工标注数据微调基座模型，得到 π_sft
Step 2: Train RM     → 用偏好数据训一个独立的 reward model r_φ(x, y)
Step 3: RL (PPO)     → 用 r_φ 当奖励信号，PPO 优化 π_sft → π_θ
```

痛点：Step 2 和 Step 3 各需要一个完整模型在显存里。70B 模型根本放不下 4 个模型（Actor + Critic + RM + Reference）。

### DPO 的核心洞察

DPO 作者（Rafailov 2023）发现：**RLHF 的 Step 3（PPO 优化）有一个 closed-form 解**。

也就是说，你不需要跑 RL 循环来优化策略。最优策略可以直接用数学公式表达出来。

推导过程（不用背，看思路就行）：

**RLHF 的优化目标是**：

$$\max_\pi \; \mathbb{E}_{x,y}[r(x,y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

翻译：最大化奖励，同时别偏离 reference model（SFT 模型）太远。

**DPO 证明了**，这个优化问题的最优解是：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x)$ 是归一化常数。

**关键一步**：把上面的公式反解出 $r(x,y)$：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**这就是 DPO 的隐式 reward model**：

$$\boxed{r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}}$$

**翻译成人话**：一个 response 的"奖励"就是——**当前策略比 reference 模型更倾向于生成这个 response 的程度**。

- 如果 $\pi_\theta$ 生成 $y$ 的概率比 $\pi_{ref}$ 高很多 → 奖励高（策略觉得这个 response 好）
- 如果 $\pi_\theta$ 和 $\pi_{ref}$ 差不多 → 奖励接近 0
- 如果 $\pi_\theta$ 比 $\pi_{ref}$ 更不想生成 $y$ → 奖励为负

---

## 算法流程

### Loss 函数

把隐式 reward 代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

用隐式 reward 替换：

$$= \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

**Loss 函数**：

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### 你只需要

1. 一个 $\pi_\theta$（正在训练的策略模型）
2. 一个 $\pi_{ref}$（SFT 模型，冻结不训）
3. 偏好数据 $(x, y_w, y_l)$

### 训练过程

对于每个偏好对，让模型更倾向于生成 $y_w$、更不倾向于生成 $y_l$。

### 一个具体例子

```
Prompt: "解释黑洞"
y_w (赢): "黑洞是时空曲率极大的区域，连光都无法逃逸。由广义相对论预言..."
y_l (输): "黑洞就是一个很黑很黑的洞，啥都吸进去。"
```

DPO 训练的一步：

1. 用 $\pi_\theta$ 算 $\log \pi_\theta(y_w|x)$ 和 $\log \pi_\theta(y_l|x)$
2. 用 $\pi_{ref}$ 算 $\log \pi_{ref}(y_w|x)$ 和 $\log \pi_{ref}(y_l|x)$
3. 算差值：$\beta \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]$
4. 过 sigmoid → 越接近 1 越好（说明策略偏好赢的那个）
5. 反向传播，更新 $\pi_\theta$

**效果**：$\pi_\theta$ 逐渐学到"生成高质量回答的概率更高，生成低质量回答的概率更低"。

### 训练数据来源

| 来源 | 方式 | 优缺点 |
|---|---|---|
| **人类标注** | 标注员看两个 response，二选一 | 质量高，但贵且慢 |
| **LLM-as-Judge** | GPT-4/Claude 代替人做判断 | 便宜快，但有偏差（倾向长回答、第一个回答）|
| **开源数据集** | Anthropic HH-RLHF、UltraFeedback 等 | 免费，但可能跟你的场景不匹配 |

---

## 优缺点

- ✅ 不需要训练 RM，不需要 PPO，不需要 RL 循环
- ✅ 训练极其简单（就是一个二分类 loss）
- ✅ 显存需求低（跟 SFT 差不多）
- ✅ 训练稳定性好
- ❌ 只用离线偏好数据，不做在线探索
- ❌ 不能动态调整奖励信号

---

## 跟传统 RLHF 的对比

| | 传统 RLHF (PPO) | DPO |
|---|---|---|
| 需要几个模型 | 4 个（Actor + Critic + RM + Reference）| **2 个**（$\pi_\theta$ + $\pi_{ref}$）|
| 训 reward model？ | ✅ 单独训一个 | ❌ 不训，策略本身暗含 |
| 需要 RL 循环？ | ✅ PPO rollout + update | ❌ 直接监督学习式训练 |
| 显存需求 | 极高 | 低（跟 SFT 差不多）|
| 训练稳定性 | 差（PPO 容易崩）| 好（就是个二分类 loss）|

---

## 演化位置

**在 RLHF 演化链中的位置**：

```
传统 RLHF (SFT → RM → PPO)
    ↓
DPO (2023) ← **你在这里**
    ↓
Online DPO / SimPO (2024-2025)
    ↓
GRPO / DAPO (2024-2025)
```

详见 [[03-流程/Offline-RL与RLHF演化史]]
