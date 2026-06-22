---
tags:
  - #inverse-rl
  - #imitation-learning
---

# IRL

> Inverse Reinforcement Learning. 从专家演示中推断奖励函数，而非手工设计奖励。

## 来源与动机

标准 RL 需要手工设计奖励函数，但很多任务的奖励难以精确表述（如自动驾驶中的"安全驾驶"、机器人操作中的"自然动作"）。IRL 反转了 RL 的问题：给定专家的示范轨迹，**推断出专家在优化什么奖励函数**。这个推断出的奖励函数可以用来训练新策略或迁移到新环境。

**开创性工作**: Ng & Russell, "Algorithms for Inverse Reinforcement Learning", ICML 2000; Abbeel & Ng, "Apprenticeship Learning via Inverse Reinforcement Learning", ICML 2004

## 核心创新

IRL 的核心思想：**假设专家行为是某个未知奖励函数下的最优（或近优）策略的结果，通过匹配专家行为的特征来反推奖励函数**。与行为克隆 (BC) 不同，IRL 不仅模仿专家做了什么，还理解专家为什么这样做——因为 IRL 恢复的是奖励函数，可以在新环境中重新求解最优策略。

## 关键公式

**特征匹配条件**（Abbeel & Ng, 2004）：学到的奖励 $r^*$ 应使专家策略的价值不低于任何其他策略：
$$\mathbb{E}_{\tau \sim \pi^*}\left[\sum_t \gamma^t \phi(s_t)\right] \geq \mathbb{E}_{\tau \sim \pi}\left[\sum_t \gamma^t \phi(s_t)\right], \quad \forall \pi$$

其中 $\phi(s)$ 是特征函数，奖励假设为特征的线性组合 $r(s) = w^\top \phi(s)$。

**[[01-原子/最大熵原理|最大熵]] IRL**（Ziebart et al., 2008）：引入[[01-原子/最大熵原理|最大熵原理]]解决奖励模糊性：
$$P(\tau | \theta) \propto \exp\left(\sum_t \theta^\top \phi(s_t, a_t)\right)$$

## 优缺点

- ✅ 可以迁移到新环境（恢复的是奖励函数而非策略）
- ✅ 比行为克隆更鲁棒，不直接受复合误差影响
- ✅ 揭示了任务的内在目标结构
- ❌ **奖励模糊性**：多个奖励函数可以解释同一组专家行为
- ❌ 计算成本高——每步推断奖励都需要求解一个完整 RL 问题
- ❌ 对专家数据质量敏感，假设专家是最优的

## 演化位置

Behavioral Cloning → **IRL** → GAIL / 偏好学习
IRL 是模仿学习的重要分支。后续的 GAIL（Generative Adversarial Imitation Learning, Ho & Ermon 2016）用对抗训练绕过显式奖励恢复，直接从专家数据学习策略。IRL 的思想也影响了 LLM 对齐中的 RLHF——从人类偏好中推断奖励模型。

## 相关算法

- [[03-流程/Offline-RL与RLHF演化史|GAIL]] — 用对抗训练绕过显式奖励恢复
- Behavioral Cloning — 更简单的模仿学习基线
- RLHF — 从偏好中学习奖励模型，思想源于 IRL
