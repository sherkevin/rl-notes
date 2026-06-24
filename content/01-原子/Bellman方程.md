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

## 逐符号解读

| 符号 | 读法 | 大白话 |
|------|------|--------|
| $V^\pi(s)$ | value of state s under policy π | **状态价值**——"在状态 $s$，按策略 $\pi$ 走，平均总分" |
| $\sum_a$ | sum over actions | 对所有可能的动作求和 |
| $\pi(a\|s)$ | policy | 在状态 $s$ 下选动作 $a$ 的**概率**（策略决定的） |
| $\sum_{s', r}$ | sum over next states and rewards | 对所有可能的下一个状态 $s'$ 和奖励 $r$ 求和 |
| $p(s', r \| s, a)$ | transition probability | 在状态 $s$ 执行动作 $a$ 后，到达 $s'$ 并拿到奖励 $r$ 的**概率**（环境决定的） |
| $r$ | reward | 执行这个动作后**立刻拿到的分数** |
| $\gamma$ | gamma, discount factor | **折扣因子**（$0 < \gamma < 1$），未来奖励打折。$\gamma = 0.99$ 表示"10 步后的 1 分 ≈ 现在的 0.9 分" |
| $V^\pi(s')$ | value of next state | 到达下一个状态 $s'$ 后，继续按 $\pi$ 走的平均总分 |
| $r + \gamma V^\pi(s')$ | | **这一步的即时奖励 + 打折后的未来价值** |

---

## 整体怎么读

$$V^\pi(s) = \underbrace{\sum_a \pi(a|s)}_{\text{策略选动作}} \underbrace{\sum_{s', r} p(s', r | s, a)}_{\text{环境给反馈}} \underbrace{[r + \gamma V^\pi(s')]}_{\text{这步得分 + 未来得分}}$$

从里往外读：

**最内层** $r + \gamma V^\pi(s')$：

```
这一局的总分 = 这一步拿的分(r) + 打折后(γ)从下一步开始继续玩的价值(V^π(s'))
```

**中间层** $\sum_{s',r} p(s',r|s,a) [\cdots]$：

```
在状态 s 执行动作 a 后，环境可能给你不同的 (s', r)
按概率加权平均 → "执行动作 a 的期望得分"
```

**最外层** $\sum_a \pi(a|s) [\cdots]$：

```
策略 π 在状态 s 下可能选不同的动作 a
按策略概率加权平均 → "在状态 s 按策略 π 走的期望总分"
```

---

## 用真例子走一遍

假设你在一个迷宫里，当前在状态 $s$（一个岔路口），策略 $\pi$ 告诉你：
- 70% 概率走**左边**（$a_1$）
- 30% 概率走**右边**（$a_2$）

**走左边**（$a_1$），环境给你两种可能：
- 80% 概率到达宝藏房（$s'_1$），奖励 $r = +10$，$V^\pi(s'_1) = 50$
- 20% 概率掉进陷阱（$s'_2$），奖励 $r = -5$，$V^\pi(s'_2) = 0$

走左边的期望得分 = $0.8 \times (10 + 0.99 \times 50) + 0.2 \times (-5 + 0.99 \times 0)$
$= 0.8 \times 59.5 + 0.2 \times (-5) = 47.6 - 1 = \mathbf{46.6}$

**走右边**（$a_2$），假设期望得分是 **20**。

**最终**：

$$V^\pi(s) = 0.7 \times 46.6 + 0.3 \times 20 = 32.62 + 6 = \mathbf{38.62}$$

---

## 一句话总结

$$V^\pi(s) = \sum_a \underbrace{\pi(a|s)}_{\text{策略选啥}} \sum_{s',r} \underbrace{p(s',r|s,a)}_{\text{环境给啥}} \cdot \underbrace{[r + \gamma V^\pi(s')]}_{\text{这步 + 以后}}$$

**当前状态的价值 = 策略选动作的概率 × 环境给反馈的概率 × （这步奖励 + 打折后的未来价值），全部求和。**

这就是 Bellman 方程的核心：**把"从 $s$ 出发的总分"拆成了"这一步的分数 + 下一步的价值"**，形成了递推关系。

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
