---
aliases: [CTDE, 集中训练分散执行, Centralized Training Decentralized Execution]
tags:
  - marl
  - ctde
  - atomic
created: 2026-06-23
---

# CTDE（集中训练，分散执行）

> MARL 的核心范式：训练时信息共享，执行时信息隔离。

---

## 一句话定义

**CTDE = 训练时所有 agent 看到全局状态，执行时每个 agent 只看到自己的观察**。

---

## 为什么需要 CTDE？

### 问题 1：完全分散（Decentralized）

每个 agent 独立学习，只看自己的观察。

**缺点**：
- 环境非平稳（其他 agent 也在学）
- 无法协调
- 收敛慢

### 问题 2：完全集中（Centralized）

一个中央控制器看到所有信息，统一决策。

**缺点**：
- 通信开销大
- 执行时不可行（agent 分散在各地）
- 单点故障

### CTDE 的解决

**训练时**：集中（用全局信息，学得快）

**执行时**：分散（只用局部信息，可部署）

---

## 具体实现

### MADDPG

$$\text{Critic: } Q_i(o_1, \ldots, o_n, a_1, \ldots, a_n)$$
$$\text{Actor: } \pi_i(o_i)$$

**训练**：Critic 用所有 agent 的观察和动作。

**执行**：Actor 只用自己的观察。

### QMIX

$$\text{Mixing: } Q_{\text{tot}}(s, a_1, \ldots, a_n) = f_{\text{mix}}(Q_1(o_1, a_1), \ldots, Q_n(o_n, a_n); s)$$

**训练**：Mixing network 用全局状态 $s$。

**执行**：每个 agent 只用自己的 $Q_i(o_i, a_i)$。

### MAPPO

$$\text{Value: } V(s_1, \ldots, s_n)$$
$$\text{Policy: } \pi_i(o_i)$$

**训练**：Value 用全局状态。

**执行**：Policy 只用局部观察。

---

## 数学形式

**集中训练**：

$$Q_i^\text{train}(o_1, \ldots, o_n, a_1, \ldots, a_n)$$

**分散执行**：

$$\pi_i^\text{exec}(o_i) = \arg\max_{a_i} Q_i^\text{train}(o_i, a_i, \text{fixed } o_{-i}, a_{-i})$$

---

## 优缺点

### 优点

- ✅ 训练效率高（信息共享）
- ✅ 可部署（执行时通信少）
- ✅ 缓解非平稳性

### 缺点

- ❌ 训练和执行的信息不对称
- ❌ 可能需要大量通信（训练时）
- ❌ 对全局状态的定义敏感

---

## 演化位置

```
Independent Q-Learning (1990s)：完全分散
    ↓
Joint Action Learners (2000s)：完全集中
    ↓
CTDE (2017, MADDPG) ← **你在这里**
    ↓
QMIX / MAPPO (2018-2022)：CTDE 的改进
```

详见 [[05-应用领域/Multi-Agent/MARL概览]]

---

## 参考资源

- 论文：Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
- 综述：Rashid et al. "Monotonic Value Function Factorisation for Deep Multi-Agent RL" (2020)
