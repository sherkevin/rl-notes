---
title: 强化学习知识库
---

# 强化学习知识库

> 从基础概念到前沿演化的完整知识体系

## 快速入口

- [[强化学习统一复习地图]] — 唯一主入口，全景、分类、演化、公式、场景选择

## 知识库结构

| 目录 | 内容 |
|---|---|
| **00-基础概念/** | MDP、回报、价值函数、Bellman 方程、TD 误差 |
| **01-分类维度/** | 按优化对象、策略类型、动作空间、数据来源等多轴分类 |
| **02-具体算法/** | DQN、PPO、SAC、GRPO 等算法的深度笔记 |
| **03-应用领域/** | 多智能体 RL、逆 RL、分层 RL |
| **04-演化综述/** | 四条演化主线的完整综述（含 2024-2026 前沿） |

## 演化主线

- [[04-演化综述/Value-Based演化史]] — Bellman → Q-Learning → DQN → Rainbow → CQL → IQL → Decision Transformer
- [[04-演化综述/Policy-AC演化史]] — REINFORCE → AC → TRPO → PPO → SAC → GRPO → DAPO
- [[04-演化综述/Model-Based演化史]] — DP → Dyna → PETS → MBPO → Dreamer → MuZero → Genie
- [[04-演化综述/Offline-RL与RLHF演化史]] — BC → CQL → IQL → DT → RLHF → DPO → GRPO → SimPO
- [[04-演化综述/MARL全景综述]] — 博弈论 → VDN → QMIX → MADDPG → MAPPO → AlphaStar → CICERO

## 最短复习路线

```
统一复习地图 → Bellman 方程 → DQN 系 → PPO/SAC 系 → Offline/RLHF 系
```
