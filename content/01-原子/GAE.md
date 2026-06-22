---
tags:
  - atomic
  - gae
  - advantage-estimation
  - bias-variance-tradeoff
created: 2026-06-22
---

# GAE（广义优势估计）

> 通过指数加权的TD残差和，在偏差和方差之间取得平衡。

---

## 核心思想

**定义**：GAE 是多个TD残差的指数加权和：

$$A^\text{GAE(\gamma, \lambda)}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是单步TD残差
- $\gamma \in [0, 1]$ 是折扣因子
- $\lambda \in [0, 1]$ 是GAE参数，控制偏差-方差权衡

---

## 展开理解

将GAE展开：

$$A^\text{GAE}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

$$= (r_t + \gamma V(s_{t+1}) - V(s_t)) + \gamma\lambda(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1})) + \cdots$$

整理后：

$$A^\text{GAE}_t = -V(s_t) + \sum_{l=0}^\infty (\gamma\lambda)^l r_{t+l} + \text{价值函数项}$$

---

## 极限情况

### $\lambda = 0$：单步TD残差

$$A^\text{GAE(\gamma, 0)}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **偏差**：高（只考虑一步）
- **方差**：低（只依赖一个奖励）

### $\lambda = 1$：蒙特卡洛估计

$$A^\text{GAE(\gamma, 1)}_t = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

- **偏差**：低（考虑完整轨迹）
- **方差**：高（依赖整个轨迹的奖励）

---

## 偏差-方差权衡

| $\lambda$ | 偏差 | 方差 | 说明 |
|----------|------|------|------|
| 0 | 高 | 低 | 单步TD，快速但粗糙 |
| 0.5 | 中 | 中 | 平衡点 |
| 0.95 | 低 | 中 | 接近MC，但仍有一定平滑 |
| 1 | 低 | 高 | 完整MC，精确但噪声大 |

**实践建议**：通常使用 $\lambda = 0.95$，在偏差和方差之间取得良好平衡。

---

## 被以下模块使用

- [[02-模块/Policy-Based/PPO]]：使用GAE作为优势估计
- [[02-模块/Policy-Based/TRPO]]：使用GAE
- [[02-模块/Actor-Critic/MAPPO]]：使用GAE

---

## 优缺点

### 优点
- ✅ 灵活控制偏差-方差权衡
- ✅ 计算高效（可以递推计算）
- ✅ 实践中效果好

### 缺点
- ❌ 需要调参 $\lambda$
- ❌ 仍然是价值函数的有偏估计

---

## 演化位置

**在优势估计演化链中的位置**：

```
TD残差 (1988)
    ↓
优势函数 (1999)
    ↓
GAE (2016) ← **你在这里**
    ↓
Retrace(λ) / V-trace (2016/2018)
```

详见 [[03-流程/Policy-AC演化史]]
