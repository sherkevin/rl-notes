---
tags:
  - rl-basics
  - single-agent
---

# Single-Agent RL

> 单智能体强化学习，区别于 Multi-Agent RL。大多数经典 RL 算法（DQN、PPO、SAC）都是单智能体的。

## 概念定义

单智能体 RL 研究**一个**智能体在**固定的**环境中如何通过与环境交互来学习最优策略。其理论基础是马尔可夫决策过程 (MDP)，形式化为五元组 $(S, A, P, R, \gamma)$。核心假设是**环境转移概率是稳定的**——智能体是环境中唯一的学习者。

## 与 Multi-Agent RL 的关键区别

| 维度 | Single-Agent RL | Multi-Agent RL |
|------|----------------|----------------|
| 环境平稳性 | 环境转移 $P(s'|s,a)$ 固定 | 其他智能体策略变化导致非平稳 |
| 理论框架 | MDP | 博弈论 / Dec-POMDP |
| 经验回放 | 可直接使用 | 旧数据可能完全失效 |
| 信用分配 | 奖励直接归因于自身动作 | 团队奖励需分解到各智能体 |
| 收敛概念 | 最优策略 $\pi^*$ | Nash 均衡（不一定是全局最优） |
| 训练范式 | 集中式 | CTDE / 分散式 |

## 核心假设

单智能体 RL 的所有理论保证（收敛性、最优性）都建立在**马尔可夫性**和**环境平稳性**之上：

$$P(s_{t+1} | s_t, a_t) \text{ 是固定的，不随时间变化}$$

一旦引入多个学习智能体，这个假设彻底崩塌——其他智能体的策略在不断更新，从智能体 $i$ 的视角看，环境变成了非平稳的。

## 经典算法族

- **Value-Based**: DQN, Rainbow, CQL, IQL
- **Policy-Based**: REINFORCE, PPO, TRPO
- **Actor-Critic**: DDPG, TD3, SAC
- **Model-Based**: Dyna-Q, PETS, Dreamer, MuZero

## 相关算法

- [[Multi-Agent]] — 多智能体场景的扩展
- [[00-基础概念/基础概念|MDP]] — 单智能体 RL 的理论基础
