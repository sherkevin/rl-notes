# 强化学习知识库

> 从基础概念到前沿演化的完整知识体系

## 📚 知识库结构

| 目录 | 内容 | 文档类型 |
|---|---|---|
| **00-基础概念/** | MDP、回报、价值函数、Bellman 方程、TD 误差、KL 散度、Reward Model | 基础概念卡片 |
| **01-分类维度/** | 按优化对象、策略类型、动作空间、数据来源等多轴分类 | 📇 索引卡片 + 详细解释 |
| **02-具体算法/** | DQN、PPO、SAC、GRPO 等算法的深度笔记 | 算法详解 |
| **03-应用领域/** | 多智能体 RL、逆 RL、分层 RL | 应用专题 |
| **04-演化综述/** | 四条演化主线的完整综述（含 2024-2026 前沿） | 演化全景 |

### 📇 索引卡片 vs 详细解释

`01-分类维度/` 下有两类文档：

- **英文命名的索引卡片**（如 `Discrete-Actions.md`、`On-Policy.md`）：自动索引，列出相关算法，顶部有 📇 标记
- **中文命名的详细解释**（如 `按动作空间.md`、`按策略类型分类.md`）：深入讲解分类维度，包含对比表和决策指南

## 🚀 快速入口

- [[强化学习统一复习地图]] — 唯一主入口，全景、分类、演化、公式、场景选择

## 📖 最短复习路线

```
统一复习地图 → Bellman 方程 → DQN 系 → PPO/SAC 系 → Offline/RLHF 系
```

## 🎯 基础专题（00-基础概念/）

- [[00-基础概念/基础概念]]：MDP、回报、价值函数等地基
- [[00-基础概念/Bellman方程和TD误差]]：公式体系的核心
- [[00-基础概念/Forward-KL与Reverse-KL]]：Forward/Reverse KL 的直觉与数学，RL 约束的核心理论
- [[00-基础概念/Reward-Model训练方法]]：Reward model 八大训练方法全景（Pairwise/Pointwise/PRM/LLM-as-Judge/Verifiable/DPO 等）

## 📊 分类维度（01-分类维度/）

按不同维度对 RL 算法进行分类：

- [[01-分类维度/按优化对象分类]]：Value-Based / Policy-Based / Actor-Critic
- [[01-分类维度/按策略类型分类]]：On-Policy / Off-Policy
- [[01-分类维度/按动作空间]]：Discrete / Continuous
- [[01-分类维度/按数据来源]]：Online / Offline
- [[01-分类维度/按模型依赖]]：Model-Free / Model-Based
- [[01-分类维度/按学习范式]]：Tabular / Function Approximation

## 🔬 深度专题（04-演化综述/）

- [[04-演化综述/Value-Based演化史]]：从 Bellman 到 Diffusion-RL，18 个方法的完整演化链
- [[04-演化综述/Policy-AC演化史]]：从 REINFORCE 到 DAPO，14 个方法 + 三条演化线（通用策略优化 / 连续控制 / LLM 对齐）
- [[04-演化综述/Model-Based演化史]]：从 DP 到 Genie/GameNGen，21 个方法 + 三条演化线（动力学模型 / 世界模型 / 搜索规划）
- [[04-演化综述/Offline-RL与RLHF演化史]]：从 BC 到 SimPO，21+ 个方法 + 两大主线（离线数据利用 / 偏好优化）
- [[04-演化综述/MARL全景综述]]：从博弈论到 CICERO，涵盖值分解、策略梯度、通信、自我博弈等 6 大板块

## 🌐 在线版本

本知识库已部署为在线网站，支持全文搜索、知识图谱、暗色模式：

https://sherkevin.github.io/rl-notes/
