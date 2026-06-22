---
tags:
  - #actor-critic
  - #off-policy
  - #offline-rl
---

# AWAC

> Advantage-Weighted Actor-Critic. 离线 RL 中用 advantage 加权策略更新，结合 BC 和 RL 的优点。

## 来源与动机

纯离线方法难以在线微调。CQL 等方法虽然有效但在线微调时表现不佳。AWAC 的目标是设计一个方法，既利用离线数据，又能在在线交互中持续改进。核心问题：如何让策略更新永远不会查询 OOD 动作，同时支持离线→在线的无缝过渡？

**论文**: Nair, Dalal, Gupta, Levine, "Accelerating Online Reinforcement Learning with Offline Datasets", 2020 (arXiv:2011.09199)

## 核心创新

用优势函数加权的行为克隆来更新策略，避免策略查询 OOD 动作。策略更新类似加权 BC，权重由 Q 函数的优势值决定。好的动作（$A > 0$）被指数加权放大，差的动作被抑制。当 $A=0$ 时退化为纯 BC，天然连接了模仿学习和强化学习。

## 关键公式

$$\mathcal{L}_\pi = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\left(\frac{A(s,a)}{\lambda}\right) \log \pi_\theta(a|s) \right]$$

其中 $A(s,a) = Q(s,a) - V(s)$ 是优势函数。好的动作（$A > 0$）被指数加权放大，差的动作被抑制。$\lambda$ 控制加权的温度。

## 优缺点

- ✅ 天然支持离线→在线微调（off-policy critic 持续更新）
- ✅ 策略更新永远不会查询 OOD 动作
- ✅ 与 BC 的联系清晰（$A=0$ 时退化为 BC）
- ❌ 仍然需要训练 critic，面临分布偏移
- ❌ 优势函数的尺度选择影响训练

## 演化位置

BCQ → CQL → **AWAC** → IQL
AWAC 是"优势加权行为克隆"这一思想的早期代表，直接影响了 IQL 的策略提取步骤和后续 AWAC-style 的离线-在线混合方法。

## 相关算法

- [[CQL]] — 保守 Q 值方法
- [[IQL]] — 继承 AWAC 策略提取步骤并进一步发展
- [[TD3+BC]] — 同期极简离线 RL 方法
- 完整演化: 见 [Offline-RL与RLHF演化史](../../04-演化综述/Offline-RL与RLHF演化史.md#a4-awac-advantage-weighted-actor-critic-2020)
