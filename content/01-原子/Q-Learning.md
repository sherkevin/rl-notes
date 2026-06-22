---
tags:
  - atomic
  - q-learning
  - off-policy
  - foundational
created: 2026-06-22
---

# Q-Learning

> 最基础的Off-policy TD控制算法。学习最优动作价值函数。

---

## 核心思想

**目标**：学习最优动作价值函数 $Q^*(s, a)$

**Bellman最优方程**：

$$Q^*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')]$$

**Q-Learning更新规则**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

**直觉**：
- $r_t + \gamma \max_a Q(s_{t+1}, a)$：TD目标（假设下一步采取最优动作）
- $Q(s_t, a_t)$：当前估计
- 更新方向：减少预测误差

---

## Off-policy特性

**关键**：Q-Learning是Off-policy的，因为：
- **行为策略**：可以是任意策略（如 $\epsilon$-greedy）
- **目标策略**：greedy策略（$\arg\max_a Q(s, a)$）

**示例**：

```python
# 行为策略：epsilon-greedy
if random() < epsilon:
    a = random_action()
else:
    a = argmax(Q[s])

# 执行动作，观察 (s, a, r, s')

# 目标策略：greedy（用于更新）
Q[s, a] += alpha * (r + gamma * max(Q[s']) - Q[s, a])
```

**优势**：可以用探索性策略收集数据，同时学习最优策略。

---

## 被以下模块使用

- [[02-模块/Value-Based/DQN]]：深度Q网络
- [[02-模块/Value-Based/Double-DQN]]：解决高估问题
- [[02-模块/Value-Based/Dueling-DQN]]：网络结构改进
- [[02-模块/Value-Based/CQL]]：离线版本

---

## 优缺点

### 优点
- ✅ **Off-policy**：可以复用历史数据
- ✅ **收敛保证**：在适当条件下收敛到最优策略
- ✅ **简单**：实现简单，易于理解

### 缺点
- ❌ **高估问题**：$\max$ 操作导致系统性高估
- ❌ **表格方法**：状态空间大时不可行
- ❌ **样本效率低**：需要大量交互

---

## 收敛条件

**定理**：如果满足以下条件，Q-Learning收敛到 $Q^*$：

1. **所有状态-动作对被无限次访问**
2. **学习率满足**：$\sum_t \alpha_t = \infty, \sum_t \alpha_t^2 < \infty$
3. **$\epsilon$-greedy的 $\epsilon$ 适当衰减**

**实践**：通常使用固定的小 $\epsilon$（如 0.01）保证持续探索。

---

## 演化位置

**在Value-Based演化链中的位置**：

```
Bellman方程 (1957)
    ↓
TD误差 (1988)
    ↓
Q-Learning (1989) ← **你在这里**
    ↓
DQN (2015)：神经网络近似
    ↓
Double DQN (2016)：解决高估
    ↓
Dueling DQN (2016)：网络结构
    ↓
Prioritized Replay (2016)：采样策略
    ↓
Rainbow (2017)：集成所有改进
```

详见 [[03-流程/Value-Based演化史]]

---

## 与SARSA的对比

| 方法 | 类型 | 更新规则 | 风险态度 |
|------|------|---------|---------|
| **Q-Learning** | Off-policy | $r + \gamma \max_a Q(s', a)$ | 乐观（假设最优） |
| **SARSA** | On-policy | $r + \gamma Q(s', a')$ | 保守（考虑探索） |

**示例**：悬崖行走问题
- Q-Learning：学习沿悬崖边走（最优但危险）
- SARSA：学习远离悬崖（安全但次优）

详见 [[02-模块/Value-Based/SARSA]]

---

## 参考资源

- 书籍：Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 6
- 论文：Watkins "Learning from Delayed Rewards" (1989)
- 代码：[OpenAI Gym Q-Learning Example](https://gymnasium.farama.org/)
