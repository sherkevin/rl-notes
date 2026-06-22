---
tags:
  - atomic
  - td-error
  - temporal-difference
  - foundational
created: 2026-06-22
---

# TD误差（时序差分误差）

> Bellman方程的采样形式，衡量预测与实际的差异。TD学习的核心。

---

## 核心思想

**定义**：TD误差是Bellman方程的采样版本：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**直觉**：
- $r_t + \gamma V(s_{t+1})$：实际观察到的回报（TD目标）
- $V(s_t)$：当前的价值估计
- $\delta_t$：预测误差

---

## 三种形式的TD误差

### 1. 状态价值的TD误差

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**用途**：更新状态价值函数 $V(s)$

### 2. 动作价值的TD误差

$$\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

**用途**：SARSA等On-policy方法

### 3. Q-Learning的TD误差

$$\delta_t = r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)$$

**用途**：Q-Learning等Off-policy方法

---

## TD学习更新规则

### TD(0) 更新

$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$

其中 $\alpha$ 是学习率。

**展开**：

$$V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]$$

**直觉**：
- 如果 $\delta_t > 0$：实际比预期好，增加 $V(s_t)$
- 如果 $\delta_t < 0$：实际比预期差，减少 $V(s_t)$

---

## 被以下模块使用

- [[02-模块/Value-Based/Q-Learning]]：Off-policy TD控制
- [[02-模块/Value-Based/SARSA]]：On-policy TD控制
- [[02-模块/Value-Based/DQN]]：深度Q网络
- [[01-原子/GAE]]：广义优势估计

---

## 优缺点

### 优点
- ✅ **无需模型**：不需要知道转移概率
- ✅ **在线学习**：每步都可以更新
- ✅ **低方差**：比蒙特卡洛方法的方差小

### 缺点
- ❌ **有偏估计**：$V(s_{t+1})$ 是估计值，引入偏差
- ❌ **Bootstrap问题**：误差会累积
- ❌ **学习率敏感**：$\alpha$ 的选择影响收敛

---

## 与蒙特卡洛方法的对比

| 方法 | 目标 | 偏差 | 方差 | 是否需要完整轨迹 |
|------|------|------|------|------------------|
| **TD(0)** | $r_t + \gamma V(s_{t+1})$ | 高 | 低 | 否 |
| **MC** | $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ | 无 | 高 | 是 |
| **TD(λ)** | 加权平均 | 中 | 中 | 可选 |

---

## 演化位置

**在TD学习演化链中的位置**：

```
Bellman方程 (1957)
    ↓
TD误差 (1988) ← **你在这里**
    ↓
TD(0) (1988)：单步更新
    ↓
TD(λ) (1988)：多步更新
    ↓
Q-Learning (1989)：Off-policy TD
    ↓
SARSA (1994)：On-policy TD
    ↓
GAE (2016)：广义优势估计
```

详见 [[03-流程/Value-Based演化史]]

---

## 与Bellman方程的关系

**[[01-原子/Bellman方程|Bellman方程]]**（期望形式）：

$$V^\pi(s) = \mathbb{E}_\pi[r + \gamma V^\pi(s') | s]$$

**TD误差**（采样形式）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**关系**：
- Bellman方程是期望值，TD误差是采样值
- 当 $V = V^\pi$ 时，$\mathbb{E}[\delta_t] = 0$
- TD学习通过最小化TD误差来逼近Bellman方程

详见 [[01-原子/Bellman方程]]

---

## 参考资源

- 书籍：Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 6
- 论文：Sutton "Learning to predict by the methods of temporal differences" (1988)
- 教程：[David Silver's RL Course Lecture 4](https://www.davidsilver.uk/teaching/)
