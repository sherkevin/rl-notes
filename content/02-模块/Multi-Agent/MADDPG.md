---
tags:
  - #multi-agent
  - #actor-critic
  - #off-policy
---

# MADDPG

> Multi-Agent [[02-模块/Actor-Critic/DDPG|DDPG]] (Lowe 2017). CTDE 范式：集中训练分散执行，每个 agent 有独立 critic。

## 来源与动机

单智能体 [[02-模块/Actor-Critic/DDPG|DDPG]] 无法直接用于多智能体环境，因为**非平稳性**问题：每个智能体面对的环境包含其他正在学习的智能体，状态转移分布持续变化，[[01-原子/经验回放|经验回放]]失效。MADDPG 将 [[02-模块/Actor-Critic/DDPG|DDPG]] 扩展到多智能体，通过 CTDE（集中训练 + 分散执行）解决非平稳性。

**论文**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017, OpenAI + DeepMind

## 核心创新

每个智能体有一个**中心化 critic** 和一个**分散式 actor**。critic 可以看到全局信息 $o_i + (o_j, a_j)_{j \neq i}$，actor 只看本地观测。这是 CTDE 范式的经典实现：训练时利用全局信息获得更稳定的学习信号，执行时只用局部观测做决策。

## 关键架构与公式

- **Actor**: $\pi_i(o_i) \to a_i$（分散式，执行时只依赖本地观测）
- **Critic**: $Q_i(s, a_1, \ldots, a_n)$（集中式，训练时使用全局信息）

**Critic 更新**：
$$L = E[(Q_i(s, a) - y)^2], \quad y = r_i + \gamma \cdot Q_i'(s', a_1', \ldots, a_n')$$

**Actor 更新**：
$$\nabla_{\theta_i} J = -\nabla_{\theta_i} Q_i(s, a_1, \ldots, \pi_i(o_i), \ldots, a_n)$$

## 优缺点

- ✅ CTDE 的经典实现，在 MPE（Particle World）环境中表现优异
- ✅ 可以处理竞争和合作设定
- ✅ 中心化 critic 缓解非平稳性问题
- ❌ 基于 [[02-模块/Actor-Critic/DDPG|DDPG]]，继承其对超参数敏感、训练不稳定的缺点
- ❌ critic 的输入维度随智能体数量线性增长，扩展性差

## 演化位置

COMA (2018) ← **MADDPG** → [[02-模块/Multi-Agent/MAPPO|MAPPO]] (2022)
MADDPG 是 CTDE + [[01-原子/策略梯度|策略梯度]]的标志性工作。[[02-模块/Multi-Agent/MAPPO|MAPPO]] 本质上是用 [[02-模块/Policy-Based/PPO|PPO]] 替换了 MADDPG 的 [[02-模块/Actor-Critic/DDPG|DDPG]] 框架，成为后续 MARL 研究的标准基线。

## 相关算法

- [[DDPG]] — 单智能体基础
- [[MAPPO]] — 用 [[02-模块/Policy-Based/PPO|PPO]] 替换 [[02-模块/Actor-Critic/DDPG|DDPG]] 框架
- [[03-流程/MARL全景综述|COMA]] — 反事实基线的[[01-原子/策略梯度|策略梯度]] MARL
- [[QMIX]] — 值分解路线的 CTDE 方法
- 完整演化: 见 [MARL全景综述](../../03-流程/MARL全景综述.md#c2-maddpg-multi-agent-deep-deterministic-policy-gradient)
