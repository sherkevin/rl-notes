# 强化学习知识库

> 从基础概念到前沿演化的完整知识体系，采用**原子-模块-流程**三层架构。

---

## 📐 架构设计

本知识库采用**原子-模块-流程**三层架构，充分利用 Obsidian 双链特性。

📖 **详细架构说明**：[[00-索引/三层架构]]

```
03-流程/ (Flow)      ← 演化路径、应用场景、决策流程
   ↓ 组合
02-模块/ (Module)    ← 完整算法、方法族
   ↓ 组合  
01-原子/ (Atomic)    ← 最小不可再分知识点

00-索引/ (Index)     ← 多维度入口
```

### 层级说明

**01-原子/ (Atomic Layer)**
- 最小不可再分的知识点
- 被多个模块复用
- 例如：[[01-原子/Bellman方程]], [[01-原子/KL散度]], [[01-原子/策略梯度]]

**02-模块/ (Module Layer)**
- 完整算法或方法族
- 由多个原子组合而成
- 例如：[[02-模块/Value-Based/DQN]], [[02-模块/Policy-Based/PPO]]

**03-流程/ (Flow Layer)**
- 演化路径：模块之间的历史演进
- 应用场景：如何根据问题选择模块
- 决策流程：SFT → RM → PPO 等

**00-索引/ (Index Layer)**
- 多维度入口页面
- 按不同视角组织知识
- 例如：Policy视角、Reward视角

---

## 🚀 快速入口

### 按视角浏览

- [[00-索引/Policy视角]] — 从策略优化角度串联知识
- [[00-索引/Reward视角]] — 从奖励建模角度串联知识
- [[强化学习统一复习地图]] — 全景、分类、演化、公式、场景选择

### 按层级浏览

**想看基础概念？** → [[01-原子/]]
- [[01-原子/Bellman方程]]、[[01-原子/TD误差]]、[[01-原子/KL散度]]
- [[01-原子/策略梯度]]、[[01-原子/优势函数]]、[[01-原子/GAE]]
- [[01-原子/Clip机制]]、[[01-原子/最大熵原理]]

**想看具体算法？** → [[02-模块/]]
- Value-Based: [[02-模块/Value-Based/DQN]]、[[02-模块/Value-Based/Q-Learning]]
- Policy-Based: [[02-模块/Policy-Based/PPO]]、[[02-模块/Policy-Based/TRPO]]
- Actor-Critic: [[02-模块/Actor-Critic/SAC]]、[[02-模块/Actor-Critic/DDPG]]
- Reward-Model: [[02-模块/Reward-Model/DPO]]、[[02-模块/Reward-Model/PRM]]

**想看演化历史？** → [[03-流程/]]
- [[03-流程/Value-Based演化史]] — 从 Bellman 到 Diffusion-RL
- [[03-流程/Policy-AC演化史]] — 从 REINFORCE 到 DAPO
- [[03-流程/Model-Based演化史]] — 从 DP 到 Genie/GameNGen
- [[03-流程/Offline-RL与RLHF演化史]] — 从 BC 到 SimPO
- [[03-流程/MARL全景综述]] — 从博弈论到 CICERO

---

## 📖 最短复习路线

```
统一复习地图 → Bellman 方程 → DQN 系 → PPO/SAC 系 → Offline/RLHF 系
```

### 详细路线

1. **基础概念** (01-原子/)
   - [[01-原子/Bellman方程]] → [[01-原子/TD误差]] → [[01-原子/Q-Learning]]
   - [[01-原子/策略梯度]] → [[01-原子/优势函数]] → [[01-原子/GAE]]
   - [[01-原子/KL散度]] → [[01-原子/Clip机制]]

2. **经典算法** (02-模块/)
   - Value-Based: [[02-模块/Value-Based/DQN]] → [[02-模块/Value-Based/CQL]]
   - Policy-Based: [[02-模块/Policy-Based/PPO]] → [[02-模块/Policy-Based/GRPO]]
   - Actor-Critic: [[02-模块/Actor-Critic/SAC]]
   - Reward-Model: [[02-模块/Reward-Model/DPO]]

3. **演化脉络** (03-流程/)
   - [[03-流程/Value-Based演化史]]
   - [[03-流程/Policy-AC演化史]]
   - [[03-流程/Offline-RL与RLHF演化史]]

---

## 🎯 基础专题（01-原子/）

### 数学基础
- [[01-原子/Bellman方程]] — 动态规划的核心方程
- [[01-原子/TD误差]] — Bellman方程的采样形式
- [[01-原子/KL散度]] — Forward/Reverse KL 的直觉与数学

### 策略优化
- [[01-原子/策略梯度]] — 直接优化策略参数
- [[01-原子/优势函数]] — 降低策略梯度方差
- [[01-原子/GAE]] — 广义优势估计，平衡偏差-方差
- [[01-原子/Clip机制]] — 限制策略更新幅度
- [[01-原子/Trust-Region]] — KL散度硬约束

### 探索与采样
- [[01-原子/最大熵原理]] — 鼓励探索，防止过早收敛
- [[01-原子/重要性采样]] — 用行为策略的数据估计目标策略
- [[01-原子/重参数化技巧]] — 让随机采样操作可微分
- [[01-原子/经验回放]] — Off-policy的核心机制
- [[01-原子/目标网络]] — 稳定训练的关键技巧

### 奖励建模
- [[01-原子/Bradley-Terry模型]] — 偏好建模的理论基础
- [[01-原子/Reward-Model训练方法]] — 八大训练方法全景

---

## 🔬 深度专题（03-流程/）

### 演化综述

- [[03-流程/Value-Based演化史]] — 从 Bellman 到 Diffusion-RL，18 个方法的完整演化链
- [[03-流程/Policy-AC演化史]] — 从 REINFORCE 到 DAPO，14 个方法 + 三条演化线
- [[03-流程/Model-Based演化史]] — 从 DP 到 Genie/GameNGen，21 个方法 + 三条演化线
- [[03-流程/Offline-RL与RLHF演化史]] — 从 BC 到 SimPO，21+ 个方法 + 两大主线
- [[03-流程/MARL全景综述]] — 从博弈论到 CICERO，涵盖值分解、策略梯度、通信、自我博弈等

### 应用场景

- [[03-流程/RLHF完整流程]] — SFT → RM → PPO/DPO 的完整流程
- [[03-流程/场景选择决策树]] — 如何根据问题选择合适的 RL 方法

---

## 📊 分类维度（00-索引/）

按不同维度对 RL 算法进行分类：

- [[00-索引/按优化对象]] — Value-Based / Policy-Based / Actor-Critic
- [[00-索引/按策略类型]] — On-Policy / Off-Policy
- [[00-索引/按动作空间]] — Discrete / Continuous
- [[00-索引/按数据来源]] — Online / Offline
- [[00-索引/按模型依赖]] — Model-Free / Model-Based

### 多视角导航

- [[00-索引/Policy视角]] — 从策略优化角度串联：策略梯度 → 优势函数 → Clip机制 → PPO → GRPO
- [[00-索引/Reward视角]] — 从奖励建模角度串联：Pairwise RM → PRM → DPO → Verifiable Reward

---

## 🌐 在线版本

本知识库已部署为在线网站，支持全文搜索、知识图谱、暗色模式：

https://sherkevin.github.io/rl-notes/

---

## 📝 文档结构说明

### 原子文档模板 (01-原子/)

```markdown
---
tags: [atomic, 相关领域]
created: YYYY-MM-DD
---

# [原子名称]

> 一句话定义

## 核心概念
[简洁解释，不超过200字]

## 数学表达
[关键公式，带解释]

## 直觉理解
[类比、例子、可视化]

## 被以下模块使用
- [[模块A]]
- [[模块B]]
```

### 模块文档模板 (02-模块/)

```markdown
---
tags: [module, 优化对象, 策略类型, 动作空间]
created: YYYY-MM-DD
---

# [模块名称]

> 一句话描述

## 组合构成
本模块由以下原子组合而成：
- [[原子A]]：[在模块中的作用]
- [[原子B]]：[在模块中的作用]
- [模块特有技术]：[详细说明]

## 核心创新
[本模块独有的贡献，不重复原子内容]

## 算法流程
[伪代码或步骤]

## 优缺点
- ✅ ...
- ❌ ...

## 演化位置
[在流程中的位置，链接到03-流程/]
```

### 流程文档模板 (03-流程/)

```markdown
---
tags: [flow, 主题]
created: YYYY-MM-DD
---

# [流程名称]

> 一句话描述这条流程

## 演化全景
[时间线或演化图]

## 各阶段详解

### 阶段1：[模块A]
- 解决的问题：...
- 核心创新：...
- 链接：[[02-模块/模块A]]

### 阶段2：[模块B]
...

## 决策指南
[如何根据场景选择模块]
```

---

## 🔄 更新日志

### 2026-06-22 — 三层架构重构

- 创建新的目录结构：00-索引/、01-原子/、02-模块/、03-流程/
- 拆分长文档：Forward-KL与Reverse-KL.md → KL散度 + DPO + 知识蒸馏
- 拆分长文档：Reward-Model训练方法.md → 8个独立RM模块
- 创建原子文档：策略梯度、优势函数、GAE、Clip机制、最大熵原理、重要性采样、重参数化技巧、Trust-Region
- 精简模块文档：DQN、PPO 删除重复内容，添加原子引用
- 创建多视角索引：Policy视角、Reward视角
- 更新 README，反映新结构

---

## 🤝 贡献指南

1. **新增原子**：如果一个概念被 3 个以上模块使用，考虑提取为原子
2. **新增模块**：完整算法或方法族，引用相关原子
3. **新增流程**：演化路径或应用场景，串联相关模块
4. **更新链接**：使用 `[[路径/文档名]]` 格式，保持双链完整

---

## 📚 参考资源

- 书籍：Sutton & Barto "Reinforcement Learning: An Introduction"
- 课程：David Silver's RL Course
- 代码：OpenAI Spinning Up
- 代码：CleanRL
