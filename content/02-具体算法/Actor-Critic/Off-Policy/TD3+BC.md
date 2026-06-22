---
tags:
  - #actor-critic
  - #off-policy
  - #offline-rl
---

# TD3+BC

> TD3 + Behavioral Cloning (Fujimoto 2021). 简单有效的离线 RL 方法，在 TD3 基础上加 BC 正则化。

## 来源与动机

CQL 等方法越来越复杂，能否用极简方法达到同等效果？Fujimoto 和 Gu 提出：在 TD3 的策略损失上加一个简单的 BC 正则化项，不需要任何复杂的保守 Q 值估计。这篇工作是对"算法复杂度 vs 实际效果"这一问题的深刻反思。

**论文**: Fujimoto, Gu, "A Minimalist Approach to Offline Reinforcement Learning", NeurIPS 2021 (arXiv:2106.06860)

## 核心创新

在 TD3 的策略损失上加一个简单的 BC 正则化项。第一项是标准策略梯度（最大化 Q 值），第二项是 BC 正则（策略不要偏离数据太远）。$\alpha$ 平衡探索与保守。整个方法极简——代码改动量仅几行。

## 关键公式

$$\mathcal{L}_\pi = -\mathbb{E}_{s \sim \mathcal{D}} \left[ Q_\theta(s, \pi_\theta(s)) \right] + \alpha \cdot \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ (\pi_\theta(s) - a)^2 \right]$$

第一项是标准策略梯度（最大化 Q 值），第二项是 BC 正则（策略不要偏离数据太远）。$\alpha$ 平衡探索与保守。

## 优缺点

- ✅ **极简**：代码改动量极小（几行）
- ✅ 性能好：在 D4RL 上接近 CQL
- ✅ 训练稳定
- ❌ BC 正则过于粗糙，无法区分哪些偏离是有益的
- ❌ 超参数 $\alpha$ 的选择依赖数据集质量
- ❌ 没有理论下界保证

## 演化位置

BCQ → CQL → **TD3+BC** → IQL
证明了"简单方法 + 正确的正则化"可以与复杂方法竞争。对"算法复杂度 vs 实际效果"的反思影响了整个领域。

## 相关算法

- [[TD3]] — 本方法的 Model-Free 基础
- [[BCQ]] — 开创离线 RL 范式
- [[CQL]] — 复杂的保守 Q 值方法
- [[IQL]] — 后续更优雅的离线 RL 方案
- 完整演化: 见 [Offline-RL与RLHF演化史](../../04-演化综述/Offline-RL与RLHF演化史.md#a5-td3bc-2021)
