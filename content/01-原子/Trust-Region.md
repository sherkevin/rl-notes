---
tags:
  - atomic
  - trust-region
  - trpo
  - kl-divergence
created: 2026-06-22
---

# Trust-Region（信任区域）

> 通过限制策略更新幅度，保证单调改进。TRPO的核心思想。

---

## 核心思想

**问题**：[[01-原子/策略梯度|策略梯度]]更新步长过大，可能导致策略性能急剧下降。

**解决方案**：限制新旧策略之间的距离，确保每次更新都在"信任区域"内。

$$\max_\theta J(\theta) \quad \text{s.t.} \quad D(\pi_\theta, \pi_{\theta_\text{old}}) \leq \delta$$

其中 $D$ 是某种距离度量（通常是KL散度），$\delta$ 是信任区域大小。

---

## TRPO的信任区域

### 目标函数

$$\max_\theta \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \hat{A}_t\right]$$

### 约束条件

$$\mathbb{E}_t\left[D_{KL}(\pi_{\theta_\text{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))\right] \leq \delta$$

其中 $\delta$ 通常取 0.01。

**直觉**：
- 目标：最大化优势加权的策略比率
- 约束：新旧策略的KL散度不超过 $\delta$

---

## 为什么KL散度有效？

### 理论保证

**定理**（TRPO论文）：如果满足以下条件，策略更新保证单调改进：

$$J(\pi_\theta) - J(\pi_{\theta_\text{old}}) \geq \frac{1}{1-\gamma} \mathbb{E}_{s \sim \pi_{\theta_\text{old}}}\left[\mathbb{E}_{a \sim \pi_\theta}\left[A^{\pi_{\theta_\text{old}}}(s, a)\right]\right] - \frac{2\gamma}{(1-\gamma)^2} \epsilon \cdot D_{KL}^{\max}(\pi_{\theta_\text{old}} \| \pi_\theta)$$

其中 $\epsilon = \max_{s,a} |A^{\pi_{\theta_\text{old}}}(s, a)|$

**含义**：只要KL散度足够小，且优势估计准确，策略性能就不会下降。

### 与PPO的对比

| 方法 | 约束方式 | 优点 | 缺点 |
|------|---------|------|------|
| **[[02-模块/Policy-Based/TRPO|TRPO]]** | KL散度硬约束 | 理论保证强 | 计算复杂 |
| **[[02-模块/Policy-Based/PPO|PPO]]-Clip** | Clip软约束 | 实现简单 | 理论保证弱 |
| **[[02-模块/Policy-Based/PPO|PPO]]-Penalty** | KL散度惩罚 | 平衡 | 需要调参 |

详见 [[01-原子/Clip机制]]

---

## 求解TRPO

### 问题形式

$$\max_\theta g^T (\theta - \theta_\text{old}) \quad \text{s.t.} \quad \frac{1}{2} (\theta - \theta_\text{old})^T H (\theta - \theta_\text{old}) \leq \delta$$

其中：
- $g = \nabla_\theta J(\theta)$ 是[[01-原子/策略梯度|策略梯度]]
- $H = \nabla_\theta^2 D_{KL}$ 是Fisher信息矩阵

### 解析解

$$\theta - \theta_\text{old} = \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g$$

**问题**：计算 $H^{-1}$ 的复杂度是 $O(n^3)$，对于深度网络不可行。

### 共轭梯度法

[[02-模块/Policy-Based/TRPO|TRPO]] 使用共轭梯度法近似求解 $H^{-1} g$，避免显式计算 $H^{-1}$。

**步骤**：
1. 用共轭梯度法求解 $H x = g$，得到 $x \approx H^{-1} g$
2. 计算步长 $\alpha = \sqrt{\frac{2\delta}{g^T x}}$
3. 更新 $\theta = \theta_\text{old} + \alpha x$

---

## 被以下模块使用

- [[02-模块/Policy-Based/TRPO]]：信任区域方法的典型应用
- [[02-模块/Policy-Based/PPO]]：简化版的信任区域
- [[02-模块/Reward-Model/DPO]]：隐式信任区域

---

## 优缺点

### 优点
- ✅ **理论保证**：保证单调改进
- ✅ **稳定性好**：不会大幅退化
- ✅ **自适应步长**：自动选择合适的更新步长

### 缺点
- ❌ **计算复杂**：需要计算Fisher信息矩阵
- ❌ **实现复杂**：需要共轭梯度法
- ❌ **样本效率低**：每次更新需要新数据

---

## 演化位置

**在策略更新约束演化链中的位置**：

```
无约束策略梯度 (高方差，不稳定)
    ↓
Trust-Region (2015)：KL散度硬约束 ← **你在这里**
    ↓
PPO-Clip (2017)：Clip软约束
    ↓
PPO-Penalty (2017)：KL散度惩罚
    ↓
GRPO (2024)：去Critic，保留Clip
```

详见 [[03-流程/Policy-AC演化史]]

---

## 与KL散度的关系

[[02-模块/Policy-Based/TRPO|TRPO]] 使用KL散度作为距离度量：

$$D_{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}}\left[\log \frac{\pi_{\theta_\text{old}}(a|s)}{\pi_\theta(a|s)}\right]$$

**为什么用Reverse KL而不是Forward KL？**

详见 [[01-原子/KL散度]]

**关键**：Reverse KL 允许新策略在某些地方给零概率（Zero-Forcing），更适合策略优化。

---

## 参考资源

- 论文：Schulman et al. "Trust Region Policy Optimization" (2015)
- 论文：Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- 代码：[OpenAI Spinning Up [[02-模块/Policy-Based/TRPO|TRPO]]](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
