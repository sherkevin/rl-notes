---
tags:
  - model-free
  - index
---

# Model-Free 算法汇总

> 汇总所有 Model-Free RL 算法，按 Value-Based、Policy-Based、Actor-Critic 分类索引。

## 概述

Model-Free RL 不学习环境模型（转移概率和奖励函数），而是通过与环境交互直接学习策略或价值函数。所有方法的核心权衡：**如何在不知道世界怎么运转的情况下，仅靠试错来学习最优行为？**

## Value-Based 方法

> 学习价值函数 $Q(s,a)$ 或 $V(s)$，策略从中隐式导出（如 $\epsilon$-greedy）。

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| Q-Learning | 1989 | Off-policy + max 操作 | [[Q-Learning]] |
| DQN | 2015 | 深度网络 + 经验回放 + 目标网络 | [[DQN]] |
| Double DQN | 2016 | 解耦选择和评估，减少高估 | [[DDQN]] |
| Dueling DQN | 2016 | V(s) 和 A(s,a) 分解 | [[02-具体算法/Value-Based/DQN|Dueling DQN]] |
| PER | 2016 | 按 TD 误差优先级采样 | [[02-具体算法/Value-Based/DQN|PER]] |
| C51 | 2017 | 分布式 RL，学习回报分布 | [[04-演化综述/Value-Based演化史|C51]] |
| QR-DQN | 2018 | 分位数回归分布 RL | [[04-演化综述/Value-Based演化史|QR-DQN]] |
| Rainbow | 2018 | DQN 六大改进的集大成者 | [[04-演化综述/Value-Based演化史|Rainbow]] |
| BCQ | 2018 | 离线 RL，约束动作到数据支撑集 | [[BCQ]] |
| CQL | 2020 | 保守 Q 值正则化 | [[CQL]] |
| IQL | 2022 | expectile 回归，不查 OOD 动作 | [[IQL]] |

**演化主线**: Q-Learning → DQN → Rainbow → CQL → IQL → Decision Transformer

## Policy-Based 方法

> 直接学习参数化策略 $\pi_\theta(a|s)$，不通过价值函数间接导出。

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| REINFORCE | 1992 | 策略梯度开山之作 | [[REINFORCE]] |
| TRPO | 2015 | KL 约束保证单调改进 | [[TRPO]] |
| PPO | 2017 | Clip 替代 KL，极简实现 | [[PPO]] |

**演化主线**: REINFORCE → Policy Gradient Theorem → Natural PG → TRPO → PPO → GRPO

## Actor-Critic 方法

> 结合策略（Actor）和价值函数（Critic），Actor 优化策略，Critic 评估好坏。

### On-Policy Actor-Critic

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| A3C/A2C | 2016 | 并行化 Actor-Critic | [[A3C]] |
| PPO | 2017 | Clip + 多 epoch 更新 | [[PPO]] |

### Off-Policy Actor-Critic（连续控制）

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| DDPG | 2016 | 确定性策略梯度 + 深度网络 | [[DDPG]] |
| TD3 | 2018 | 双 Critic + 延迟更新 + 目标平滑 | [[TD3]] |
| SAC | 2018 | 最大熵 + 自动温度调节 | [[SAC]] |

### Offline Actor-Critic

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| AWAC | 2020 | 优势加权行为克隆 | [[AWAC]] |
| TD3+BC | 2021 | 极简 BC 正则化 | [[TD3+BC]] |

## 序列建模范式（后 Bellman）

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| Decision Transformer | 2021 | RL = 条件序列建模 | [[04-演化综述/Offline-RL与RLHF演化史|Decision Transformer]] |

## LLM 对齐方法

| 算法 | 年份 | 关键贡献 | 链接 |
|------|------|---------|------|
| DPO | 2023 | 直接偏好优化，跳过奖励模型 | [[04-演化综述/Offline-RL与RLHF演化史|DPO]] |
| GRPO | 2024 | 组内归一化替代 Critic | [[GRPO]] |
| DAPO | 2025 | 解耦 clip + 动态采样 | [[DAPO]] |

## 相关算法

- [Value-Based 演化史](../04-演化综述/Value-Based演化史.md)
- [Policy-AC 演化史](../04-演化综述/Policy-AC演化史.md)
- [Offline-RL 与 RLHF 演化史](../04-演化综述/Offline-RL与RLHF演化史.md)
- [[📋 Model-Based算法汇总]]
- [[📋 Policy-Based算法汇总]]
