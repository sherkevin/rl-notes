---
aliases: [MARL, Multi-Agent RL, 多智能体强化学习]
tags:
  - marl
  - multi-agent
  - overview
created: 2026-06-23
---

# Multi-Agent RL（多智能体强化学习）

> 多个智能体在共享环境中协作或竞争。从博弈论到 AlphaStar 到 CICERO。

---

## 一句话定义

**MARL = 多个 agent 在同一个环境里交互，每个 agent 学自己的策略**。

与单 agent RL 的区别：环境不再是静态的——其他 agent 也在学，所以"环境"在变。

---

## 核心挑战

### 1. 非平稳性（Non-Stationarity）

**问题**：其他 agent 也在学，所以环境在变。

$$p(s' | s, a_i) \text{ 不再是固定的} \implies \text{单 agent RL 的收敛性保证失效}$$

**解决**：
- 经验回放（固定历史数据）
- 集中训练分散执行（CTDE）
- 对手建模

### 2. 信用分配（Credit Assignment）

**问题**：团队赢了，谁的贡献大？

$$R = \sum_i r_i \quad \text{但每个 } r_i \text{ 怎么分？}$$

**解决**：
- 差分奖励（difference reward）
- Shapley value
- Counterfactual regret

### 3. 维度灾难

**问题**：状态空间和动作空间随 agent 数量指数增长。

$$|\mathcal{S}| = |\mathcal{S}_1| \times |\mathcal{S}_2| \times \cdots \times |\mathcal{S}_n|$$

**解决**：
- 局部观察（每个 agent 只看自己的观察）
- 分解方法（QMIX、VDN）
- 通信机制

---

## 三大范式

### 1. Fully Cooperative（完全合作）

**目标**：所有 agent 最大化同一个全局奖励 $R$。

**代表算法**：
- [[05-应用领域/Multi-Agent/QMIX|QMIX]]：分解 Q 函数
- [[05-应用领域/Multi-Agent/MAPPO|MAPPO]]：集中训练 + 分散执行
- VDN：值函数分解

**应用**：
- 机器人协作
- 自动驾驶车队
- 星际争霸（AlphaStar）

---

### 2. Fully Competitive（完全竞争）

**目标**：零和博弈，$R_1 = -R_2$。

**代表算法**：
- Self-play（自我对弈）
- PSRO（策略空间响应先知）
- CFR（反事实遗憾最小化）

**应用**：
- 围棋（AlphaGo）
- 扑克（Pluribus、DeepStack）
- 电子竞技

---

### 3. Mixed Cooperative-Competitive（混合）

**目标**：部分合作、部分竞争（一般和博弈）。

**代表算法**：
- [[05-应用领域/Multi-Agent/MADDPG|MADDPG]]：集中训练 + 分散执行
- Nash Q-Learning
- Mean Field RL

**应用**：
- 谈判（CICERO）
- 拍卖
- 交通流优化

---

## CTDE：集中训练，分散执行

**核心思想**：训练时可以用全局信息，执行时只用局部观察。

```
训练阶段：
  Q_i(o_i, a_i, o_{-i}, a_{-i})  ← 用所有 agent 的观察和动作

执行阶段：
  π_i(o_i)  ← 只用 agent i 自己的观察
```

**为什么这样做？**
- 训练时有全局信息，学得更好
- 执行时通信受限，只能局部决策
- 解耦训练和执行的信息需求

**代表算法**：
- [[05-应用领域/Multi-Agent/MADDPG|MADDPG]]：每个 agent 有自己的 critic（用全局信息）
- [[05-应用领域/Multi-Agent/QMIX|QMIX]]：集中 mixing network + 分散 Q 网络
- [[05-应用领域/Multi-Agent/MAPPO|MAPPO]]：集中 value + 分散 policy

---

## 信用分配方法

### 1. Difference Reward

$$D_i(s, a_i) = R(s, a_i, a_{-i}) - R(s, a_i^0, a_{-i})$$

其中 $a_i^0$ 是"什么都不做"的动作。

**直觉**：agent $i$ 的贡献 = 有它 vs 没它。

### 2. Shapley Value

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

**直觉**：agent $i$ 对所有可能联盟的边际贡献的加权平均。

**优点**：公平、唯一满足某些公理。

**缺点**：计算复杂（$O(2^n)$）。

### 3. Counterfactual Multi-Agent（COMA）

$$A_i(s, a_i) = Q(s, a_i, a_{-i}) - \sum_{a_i'} \pi_i(a_i' | o_i) Q(s, a_i', a_{-i})$$

**直觉**：agent $i$ 的实际动作 vs 策略的期望动作。

---

## 演化位置

```
博弈论 (1944, von Neumann)
    ↓
Stochastic Games (1953, Shapley)
    ↓
Minimax Q-Learning (1994)
    ↓
Nash Q-Learning (2003)
    ↓
MADDPG (2017)：CTDE
    ↓
QMIX (2018)：值函数分解
    ↓
MAPPO (2022)：集中训练 + 分散执行
    ↓
CICERO (2022)：人类水平谈判
```

详见 [[03-流程/MARL全景综述]]

---

## 参考资源

- 综述：Albrecht et al. "Multi-Agent Reinforcement Learning: A Selective Overview" (2021)
- 论文：Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017, MADDPG)
- 论文：Rashid et al. "QMIX: Monotonic Value Function Factorisation" (2018)
- 论文：Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2022, MAPPO)
- 论文：Bakhtin et al. "Human-Level Play in the Game of Diplomacy" (2022, CICERO)
