---
tags:
  - atomic
  - bellman-equation
  - dynamic-programming
  - foundational
aliases:
  - Bellman方程
  - 贝尔曼方程
created: 2026-06-22
---

# Bellman方程

> 价值函数的递推关系。将复杂问题分解为子问题。动态规划和RL的理论基础。

---

## 核心思想

**状态价值函数的Bellman方程**：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$$

**直觉**：
- 当前状态的价值 = 即时奖励 + 折扣后的下一状态价值
- 这是一个递推关系，可以用动态规划求解

---

## 两种形式

### 1. Bellman期望方程（策略评估）

$$V^\pi(s) = \mathbb{E}_\pi[r + \gamma V^\pi(s') | s]$$

**展开形式**：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$$

**用途**：给定策略 $\pi$，计算其价值函数。

### 2. Bellman最优方程（最优策略）

$$V^*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]$$

**Q函数形式**：

$$Q^*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')]$$

**用途**：求解最优策略。

---

## 动态规划求解

### 策略评估（Policy Evaluation）

**迭代更新**：

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$$

**收敛**：当 $k \to \infty$，$V_k \to V^\pi$

### 策略改进（Policy Improvement）

**贪心策略**：

$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$$

**保证**：$\pi'$ 至少和 $\pi$ 一样好

### 策略迭代（Policy Iteration）

交替执行策略评估和策略改进，直到收敛。

### 价值迭代（Value Iteration）

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$$

直接逼近最优价值函数。

---

## 被以下模块使用

- [[02-模块/Value-Based/Q-Learning]]：学习最优Q函数
- [[02-模块/Value-Based/DQN]]：深度Q网络
- [[02-模块/Value-Based/SARSA]]：On-policy TD控制
- [[01-原子/TD误差]]：Bellman方程的采样形式

---

## 优缺点

### 优点
- ✅ **理论基础**：RL和动态规划的核心
- ✅ **递推关系**：可以用动态规划求解
- ✅ **收敛保证**：在适当条件下保证收敛

### 缺点
- ❌ **需要模型**：需要知道转移概率 $p(s', r | s, a)$
- ❌ **计算复杂**：状态空间大时不可行
- ❌ **维度灾难**：$|S|$ 大时无法求解

---

## 演化位置

**在RL理论基础演化链中的位置**：

```
Markov决策过程 (1950s)
    ↓
Bellman方程 (1957) ← **你在这里**
    ↓
动态规划 (1960s)
    ↓
TD学习 (1988)：无需模型的Bellman方程
    ↓
Q-Learning (1989)：Off-policy TD控制
```

详见 [[03-流程/Value-Based演化史]]

---

## 与TD误差的关系

**Bellman方程**（期望形式）：

$$V^\pi(s) = \mathbb{E}_\pi[r + \gamma V^\pi(s') | s]$$

**[[01-原子/TD误差|TD误差]]**（采样形式）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**关系**：
- Bellman方程是期望值，TD误差是采样值
- 当 $V = V^\pi$ 时，$\mathbb{E}[\delta_t] = 0$
- TD学习通过最小化TD误差来逼近Bellman方程

详见 [[01-原子/TD误差]]

---

## 参考资源

- 书籍：Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 3-4
- 论文：Bellman "Dynamic Programming" (1957)
- 教程：[David Silver's RL Course Lecture 2-3](https://www.davidsilver.uk/teaching/)
