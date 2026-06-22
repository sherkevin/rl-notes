---
tags:
  - #value-based
  - #off-policy
  - #offline-rl
---

# IQL

> Implicit Q-Learning (Kostrikov 2022). 通过 expectile 回归避免查询 OOD 动作的 Q 值，离线 RL SOTA 之一。

## 来源与动机

CQL 和 TD3+BC 仍然需要在策略优化时查询 $Q(s, \pi(s))$，当 $\pi(s)$ 偏离数据分布时，Q 值不可靠。CQL 在训练过程中仍需查询 OOD 动作的 Q 值（通过 $\log \sum_a \exp Q$ 项），计算成本高且可能引入估计误差。核心问题：**能否完全不查询 OOD 动作的 Q 值，但仍然实现策略改进？**

**论文**: Kostrikov, Nair, Levine, "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022 (arXiv:2110.06169)

## 核心创新

通过 expectile 回归隐式地学习最优动作的价值，而不需要显式地知道最优动作是什么。关键洞察：把状态价值 $V(s)$ 当作随机变量（随机性来自数据集中不同动作的 Q 值），取上 expectile（接近 max）就是在不指定具体动作的情况下估计"数据中最好的动作值多少"。然后用 $V(s)$ 而非 $\max_a Q(s,a)$ 来做 Bellman 备份，完全避免了 OOD 查询。策略提取通过优势加权 BC 完成。

## 关键公式（三步）

**第一步 — expectile 回归学 $V(s)$**：
$$\mathcal{L}_V = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_2^\tau (Q(s,a) - V(s)) \right]$$
其中 $L_2^\tau(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2$。$\tau > 0.5$ 时 $V(s)$ 偏大，接近 $\max_a Q(s,a)$ 但不需要知道是哪个 $a$。

**第二步 — 用 $V(s)$ 回溯更新 $Q(s,a)$**：
$$\mathcal{L}_Q = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q(s,a) - r - \gamma V(s'))^2 \right]$$
Bellman 目标用的是 $V(s')$，不是 $\max_{a'} Q(s',a')$，完全不查询 OOD 动作。

**第三步 — 优势加权 BC 提取策略**：
$$\mathcal{L}_\pi = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp(\beta (Q(s,a) - V(s))) \log \pi_\theta(a|s) \right]$$

## 优缺点

- ✅ **彻底避免 OOD 查询**：整个训练过程 Q 函数只在数据内的 (s,a) 上被查询
- ✅ D4RL 上达到 SOTA，尤其在 AntMaze 等困难任务上
- ✅ 支持 offline-to-online 微调
- ❌ expectile 超参数 $\tau$ 需要调优
- ❌ 策略提取用 Gaussian AWAC 可能不够表达多模态分布
- ❌ 对数据集质量仍敏感

## 演化位置

BCQ → CQL → TD3+BC → **IQL** → Decision Transformer / Diffusion-RL
IQL 是离线 RL "不查询 OOD 动作" 思路的终极实现，代表了 value-based offline RL 的成熟形态。

## 相关算法

- [[BCQ]] — 开创离线 RL 的 VAE 约束方法
- [[CQL]] — 保守 Q 值正则化
- [[TD3+BC]] — 极简 BC 正则化方法
- [[AWAC]] — IQL 策略提取步骤的直接前身
- 完整演化: 见 [Offline-RL与RLHF演化史](../../03-流程/Offline-RL与RLHF演化史.md#a6-iql-implicit-q-learning-2022)
