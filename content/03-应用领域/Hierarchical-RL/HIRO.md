---
tags:
  - #hierarchical-rl
---

# HIRO

> Nachum 2018. Hierarchical RL with data re-labeling，解决分层策略的数据效率问题。

## 来源与动机

分层 RL（如 FeUdal Networks）面临一个独特的数据效率问题：高层策略生成的子目标在训练过程中不断变化，导致早期收集的经验数据中记录的子目标与当前高层策略不一致——旧数据变得"过时"。HIRO 通过数据重新标记 (data re-labeling) 解决了这个问题。

**论文**: Nachum, Gu, Lee, Levine, "Data-Efficient Hierarchical Reinforcement Learning", NeurIPS 2018, Google Brain

## 核心创新

**数据重新标记 (Data Re-labeling)**：在回放旧经验时，用当前高层策略重新计算子目标，替代原始记录的子目标。这使得旧数据可以被重新利用，大幅提升样本效率。

**两层级架构**：
- **高层策略**：每 $c$ 步输出子目标（通常是状态空间中的目标状态）
- **底层策略**：在 $c$ 步内以子目标为条件执行具体动作

## 关键公式

**高层策略更新**（用重新标记的数据）：
$$\nabla_\phi J_H = \mathbb{E}_{(s_t, g_t, R_t, s_{t+c})} \left[ \nabla_\phi \log \pi_H(g_t | s_t) \cdot A_H(s_t, g_t) \right]$$

其中 $g_t$ 是用当前高层策略重新计算的目标（而非原始记录的目标），$A_H$ 是高层优势函数。

**底层奖励**：
$$r_t^L = -\| s_{t+1} - (s_t + g_t) \|$$

即惩罚底层策略未能实现子目标指定的状态转移。

## 优缺点

- ✅ 数据重新标记大幅提升样本效率
- ✅ 在连续控制任务（如 Ant、Pusher）上有效
- ✅ 子目标用状态空间表示，比 FeUdal 的隐式目标更可解释
- ❌ 重新标记引入近似误差
- ❌ 高层和底层策略的联合优化仍然具有挑战
- ❌ 子目标空间需要与环境状态空间对齐

## 演化位置

FeUdal Networks → **HIRO** → FuN / 分层 Transformer
HIRO 是深度学习时代分层 RL 的重要进展，其数据重新标记思想被后续多个工作继承。

## 相关算法

- [[FeUdal-Networks]] — Manager-Worker 分层架构的前身
- Options Framework — 经典分层 RL 理论框架
- 分层 RL 相关方法未在演化综述中详细覆盖，属于 RL 的重要分支方向
