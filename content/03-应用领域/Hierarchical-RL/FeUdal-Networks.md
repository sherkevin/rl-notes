---
tags:
  - #hierarchical-rl
---

# FeUdal Networks

> Vezhnevets 2017. 分层 RL：Manager 设定子目标，Worker 执行具体动作。

## 来源与动机

长 horizon 任务中，扁平策略面临信用分配困难和探索效率低的问题。受中世纪封建制度启发，FeUdal Networks 将决策分解为两层：高层 Manager 负责设定长期方向（子目标），底层 Worker 负责执行具体动作。Manager 在粗时间尺度上运作，Worker 在细时间尺度上运作。

**论文**: Vezhnevets et al., "FeUdal Networks for Hierarchical Reinforcement Learning", ICML 2017, DeepMind

## 核心创新

**Manager-Worker 分层架构**：
- **Manager**：每 $c$ 步（如 10 步）输出一个子目标向量 $g_t$，指导 Worker 的行为方向
- **Worker**：每步接收 Manager 的子目标 $g_t$ 和当前观测，输出具体动作
- Manager 的奖励设计鼓励子目标与状态变化方向一致：$r_t^M = d(s_{t+c}, s_t) \cdot \cos(g_t, s_{t+c} - s_t)$

## 关键公式

**Manager 的目标**：
$$r_t^M = d(s_{t+c}, s_t) \cdot \cos(g_t, s_{t+c} - s_t)$$

即子目标 $g_t$ 应该指向 $c$ 步后的状态变化方向，且状态变化距离 $d$ 越大越好。

**Worker 的目标**：
$$r_t^W = r_t^{\text{env}} + A_t^M \cdot \cos(s_{t+1} - s_t, g_t)$$

Worker 不仅优化环境奖励，还奖励与 Manager 子目标方向一致的状态转移。

## 优缺点

- ✅ 自动发现层次结构，无需手工设计子目标
- ✅ 在 Atari Montezuma's Revenge 等长 horizon 任务上有效
- ✅ Manager 的粗时间尺度降低了长程信用分配难度
- ❌ Manager 和 Worker 的联合训练不稳定
- ❌ 子目标空间的设计需要领域知识
- ❌ 只在 Atari 上验证，连续控制任务的泛化性未知

## 演化位置

Options Framework → **FeUdal Networks** → HIRO / HAC
FeUdal Networks 是深度学习时代分层 RL 的开创性工作之一。后续的 HIRO 改进了其数据效率和子目标表示方式。

## 相关算法

- [[HIRO]] — 改进数据效率和子目标表示
- Options Framework — 经典分层 RL 理论框架
- 分层 RL 相关方法未在演化综述中详细覆盖，属于 RL 的重要分支方向
