---
tags:
  - #multi-agent
  - #actor-critic
  - #on-policy
---

# MAPPO

> Multi-Agent PPO (Yu 2022). 将 PPO 扩展到多智能体场景，性能出奇地好，是 MARL 的强基线。

## 来源与动机

社区普遍认为 on-policy 方法的样本效率不如 off-policy（如 QMIX），且 MARL 领域倾向于设计越来越复杂的算法（值分解、反事实基线等）。Yu et al. 的论文标题直接说明了一切："The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"——简单的 PPO 加上工程最佳实践就能达到甚至超过复杂方法。

**论文**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", NeurIPS 2022 (Datasets and Benchmarks Track), UC Berkeley + 清华

## 核心创新

把 PPO 应用到多智能体，配合一系列工程最佳实践，就能达到甚至超过复杂的 off-policy 方法。关键发现：PPO 在 MARL 中被低估了。通过 value normalization、advantage normalization、大批量训练、正交初始化、critic 使用全局状态等技巧，MAPPO 成为事实上的标准基线。

## 关键公式

每个智能体 $i$ 维护 actor $\pi_i$ 和 critic $V_i$：

**Actor loss (PPO clip)**：
$$L_{\text{actor}} = -E[\min(\text{ratio}_i \cdot A_i, \text{clip}(\text{ratio}_i, 1-\epsilon, 1+\epsilon) \cdot A_i)]$$

**Critic loss**：
$$L_{\text{critic}} = E[(V_i(s) - R_i)^2]$$

其中 $\text{ratio}_i = \pi_i(a_i|o_i) / \pi_i^{\text{old}}(a_i|o_i)$。

## 优缺点

- ✅ 简单、通用、稳定，不需要值分解的结构性约束
- ✅ 在 SMAC、MPE、Google Research Football、Hanabi 四个 benchmark 上表现强劲
- ✅ 成为事实上的标准基线
- ❌ 样本效率低于 off-policy 方法（on-policy 固有限制）
- ❌ 在某些需要精细信用分配的任务上不如 QMIX

## 演化位置

MADDPG → **MAPPO** → IPPO / HAPPO / MAT
MAPPO 挑战了社区的"复杂度偏见"——简单的方法加上好的工程实践可以匹敌复杂方法。成为后续 MARL 研究的标准基线。

## 相关算法

- [[PPO]] — 单智能体基础
- [[MADDPG]] — CTDE + DDPG 的前身
- [[QMIX]] — 值分解路线的对照方法
- [[04-演化综述/MARL全景综述|IPPO]] — 完全独立的 PPO，更极端的简单基线
- 完整演化: 见 [MARL全景综述](../../04-演化综述/MARL全景综述.md#c3-mappo-multi-agent-ppo)
