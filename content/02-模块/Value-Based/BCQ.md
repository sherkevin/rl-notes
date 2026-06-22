---
tags:
  - #value-based
  - #off-policy
  - #offline-rl
---

# BCQ

> Batch-Constrained deep Q-Learning (Fujimoto 2018). 限制动作空间在数据集支持范围内，解决离线 RL 的分布偏移问题。

## 来源与动机

标准 off-policy RL（DQN/DDPG）在固定 batch 数据上会灾难性失败。核心原因是**外推误差 (extrapolation error)**：Q 函数对数据集中未出现的 (s,a) 对产生严重的过高估计，而策略优化倾向于选择这些被高估的动作，形成恶性循环。BCQ 是第一个成功实现连续控制离线深度 RL 的方法，开创了"约束动作到数据集支撑集"这一范式。

**论文**: Fujimoto, Meger, Precup, "Off-Policy Deep Reinforcement Learning without Exploration", ICML 2019 (arXiv:1812.02900)

## 核心创新

约束动作空间到数据集支撑集 (support) 附近。具体做法是训练一个 VAE 来建模数据集中的动作分布，在策略优化时只考虑 VAE 能生成的动作。策略不是自由优化，而是从生成模型的候选中选最优——从根本上避免了 OOD 动作查询。

## 关键公式

$$\pi(s) = \arg\max_{a_i \sim G_\omega(s)} Q_\theta(s, a_i)$$

其中 $G_\omega$ 是生成模型 (VAE)，只生成数据集中出现过的动作类型。策略在 VAE 的候选动作中选 Q 值最大的。

## 优缺点

- ✅ 第一个成功实现连续控制离线深度 RL 的方法
- ✅ 从根本上避免了 OOD 动作查询
- ❌ VAE 训练不稳定，生成质量受限
- ❌ 动作候选离散化导致精度损失
- ❌ 对数据集覆盖度要求高

## 演化位置

Behavioral Cloning → **BCQ** → CQL → IQL → Decision Transformer
BCQ 开创了"约束动作到数据集支撑集"这一离线 RL 范式，直接启发了后续的 CQL、TD3+BC 等方法。

## 相关算法

- [[CQL]] — 用正则化项替代 VAE 约束，更简洁
- [[TD3+BC]] — 极简方法，用 BC 正则化替代 VAE
- [[IQL]] — 彻底避免 OOD 查询的终极方案
- 完整演化 (BC→BCQ→CQL→IQL→DT): 见 [Offline-RL与RLHF演化史](../../04-演化综述/Offline-RL与RLHF演化史.md#a2-bcq-batch-constrained-deep-q-learning-2018)
