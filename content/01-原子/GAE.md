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

$$A^{\text{GAE}(\gamma, \lambda)}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是单步TD残差
- $\gamma \in [0, 1]$ 是折扣因子
- $\lambda \in [0, 1]$ 是GAE参数，控制偏差-方差权衡

---

## 为什么需要 GAE？

优势函数 $A(s, a) = Q(s, a) - V(s)$ 表示"这个动作比平均水平好多少"。但真实的 $Q$ 和 $V$ 算不出来，只能**估计**。

### 两种极端估计方法

**方法 1：蒙特卡洛（MC）——跑完一整局再回头看**

$$\hat{A}_t^{\text{MC}} = \underbrace{r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} r_T}_{\text{实际拿到的总分 } G_t} - V(s_t)$$

- ✅ **没偏差**：这就是真实分数
- ❌ **方差极大**：同样的动作，有时候运气好拿了高分，有时候运气差拿了低分

**方法 2：单步 TD——只看一步就靠 $V$ 估计剩下的**

$$\hat{A}_t^{\text{TD}} = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- ✅ **方差小**：只依赖一个随机奖励 $r_t$
- ❌ **偏差大**：$V(s_{t+1})$ 本身是估计值，可能估错

### GAE 的解决：把两种方法混合

把多步 TD 估计加权平均，用 $\lambda$ 控制混合比例：

$$A^{\text{GAE}}_t = \underbrace{1 \cdot \delta_t}_{\text{第 } t \text{ 步的 TD}} + \underbrace{\gamma\lambda \cdot \delta_{t+1}}_{\text{第 } t+1 \text{ 步，打折}} + \underbrace{(\gamma\lambda)^2 \cdot \delta_{t+2}}_{\text{第 } t+2 \text{ 步，更打折}} + \cdots$$

**直觉**：每一步的 TD 误差 $\delta$ 都在说"这一步比预期好/差多少"。GAE 把这些"每一步的意外"加起来——越远的步权重越小（因为 $(\gamma\lambda)^l$ 随 $l$ 衰减）。

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

$$A^{\text{GAE}(\gamma, 0)}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **偏差**：高（只考虑一步）
- **方差**：低（只依赖一个奖励）

### $\lambda = 1$：蒙特卡洛估计

$$A^{\text{GAE}(\gamma, 1)}_t = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

- **偏差**：低（考虑完整轨迹）
- **方差**：高（依赖整个轨迹的奖励）

**为什么 $\lambda = 1$ 等于蒙特卡洛？** 把所有 $\delta$ 展开并消项（telescoping）：

$$\delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \cdots$$
$$= (r_t + \gamma V(s_{t+1}) - V(s_t)) + \gamma(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1})) + \cdots$$
$$= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t)$$
$$= G_t - V(s_t)$$

所有 $V$ 项互相消掉了！

---

## 用数字走一遍

假设一条轨迹：$s_0 \to s_1 \to s_2 \to s_3$（终止），$\gamma = 1$，$\lambda = 0.5$。

| 步 | $r_t$ | $V(s_t)$ |
|----|-------|----------|
| $t=0$ | +1 | 5 |
| $t=1$ | +2 | 4 |
| $t=2$ | +3 | 3 |
| $t=3$ | +10 | 0（终止） |

**先算每步的 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$**：

| 步 | 计算 | $\delta_t$ |
|----|------|-----------|
| $t=0$ | $1 + 1 \times 4 - 5$ | **0** |
| $t=1$ | $2 + 1 \times 3 - 4$ | **1** |
| $t=2$ | $3 + 1 \times 0 - 3$ | **0** |

**算 $A^{\text{GAE}}_0$**（$\gamma=1, \lambda=0.5$）：

$$A^{\text{GAE}}_0 = 1 \cdot 0 + 0.5 \cdot 1 + 0.25 \cdot 0 = \mathbf{0.5}$$

**对比两种极端**：

| 方法 | 计算 | 结果 |
|------|------|------|
| 单步 TD（$\lambda=0$） | $\delta_0 = 0$ | **0** |
| 蒙特卡洛（$\lambda=1$） | $G_0 - V(s_0) = (1+2+3+10) - 5$ | **11** |
| GAE（$\lambda=0.5$） | 加权混合 | **0.5** |

GAE 给了一个折中：不像 TD 那样完全忽略后面（0），也不像 MC 那样被最后的大奖励 10 拉得很高（11）。

---

## 偏差-方差权衡

| $\lambda$ | 偏差 | 方差 | 说明 |
|----------|------|------|------|
| 0 | 高 | 低 | 单步TD，快速但粗糙 |
| 0.5 | 中 | 中 | 平衡点 |
| 0.95 | 低 | 中 | 接近MC，但仍有一定平滑 |
| 1 | 低 | 高 | 完整MC，精确但噪声大 |

**实践建议**：通常使用 $\lambda = 0.95$，在偏差和方差之间取得良好平衡。

**为什么 $\lambda = 0.95$ 而不是 1？**

$V$ 网络虽然不完美，但它**大部分时候估的方向是对的**。$\lambda = 0.95$ 的意思是：

> "每多看一步，我对实际奖励的信任不变，但对 $V$ 的依赖以 $0.95^l$ 衰减。到第 20 步时，$V$ 的权重只剩 $0.95^{20} \approx 0.36$——我主要相信实际奖励，但还是稍微参考一下 $V$。"

这比 $\lambda = 1$（完全不信 $V$）方差小很多，又比 $\lambda = 0$（只信一步 $V$）偏差小很多。

---

## 一句话总结

**GAE = 把未来每一步的"意外"（TD 误差 $\delta$）用递减的权重加起来**。$\lambda$ 控制你愿意看多远——0 只看一步，1 看完整局，0.95 是实践中最好的平衡点。

---

## 被以下模块使用

- [[02-模块/Policy-Based/PPO]]：使用GAE作为优势估计
- [[02-模块/Policy-Based/TRPO]]：使用GAE
- [[05-应用领域/Multi-Agent/MAPPO]]：使用GAE

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
