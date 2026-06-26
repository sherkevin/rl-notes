# 强化学习知识库

> 从基础概念到前沿应用的完整知识体系

---

## 📐 架构设计

本知识库采用**双维度**组织方式：

### 维度一：知识深度（原子→模块→流程）

```
03-流程/ (Flow)      ← 演化路径、历史脉络
   ↓ 组合
02-模块/ (Module)    ← 完整算法、方法族
   ↓ 组合  
01-原子/ (Atomic)    ← 最小知识点、基础概念
```

### 维度二：应用领域

```
05-应用领域/ (Applications)
├── Post-Training/     ← LLM 后训练（RLHF、DPO、GRPO）
├── Multi-Agent/       ← 多智能体系统（MARL、MADDPG、QMIX）
├── Mid-Training/      ← LLM 中期训练（Self-Play、Curriculum）
└── 娱乐应用/          ← 虚拟角色、图像生成、游戏 AI
```

### 维度三：LLM 训练全流程（重点推荐）

```
06-LLM训练全流程/ (LLM Training Pipeline)
├── 01-Pre-Training阶段     ← 课程学习、数据选择 RL
├── 02-Mid-Training阶段     ← 自我对弈、推理 RL、过程奖励
└── 03-Post-Training阶段    ← RLHF、DPO、GRPO、Constitutional AI
```

**这是知识库的核心焦点**：从 Pre-Training 到 Post-Training，系统介绍 LLM 训练中 RL 技术的完整应用。

### 导航入口

```
00-索引/ (Index)       ← 多维度入口、分类视图
```

📖 **详细架构说明**：[[00-索引/三层架构]]

---

## 🚀 快速入口

### 我是新手，从哪开始？

**最短复习路线**：
```
强化学习统一复习地图 → Bellman 方程 → DQN → PPO → DPO
```

### 我关注 LLM 训练（推荐）

**LLM 全流程路线**：
```
06-LLM训练全流程/README → Pre-Training → Mid-Training → Post-Training
```

这是最实用的路线，涵盖了 LLM 训练中所有 RL 技术的应用。

**详细路线**：
1. 先读 [[强化学习统一复习地图]]，建立全景认知
2. 学习基础概念：[[01-原子/Bellman方程]] → [[01-原子/TD误差]] → [[01-原子/策略梯度]]
3. 学习经典算法：[[02-模块/Value-Based/DQN]] → [[02-模块/Policy-Based/PPO]]
4. 了解演化脉络：[[03-流程/Policy-AC演化史]]

### 我想查某个概念

- **数学符号**：[[01-原子/数学符号表]]
- **基础概念**：[[01-原子/]]
- **具体算法**：[[02-模块/]]
- **应用场景**：[[05-应用领域/]]

### 我想了解演化历史

- [[03-流程/Value-Based演化史]] — 从 Bellman 到 Diffusion-RL
- [[03-流程/Policy-AC演化史]] — 从 REINFORCE 到 DAPO
- [[03-流程/Offline-RL与RLHF演化史]] — 从 BC 到 SimPO
- [[03-流程/Model-Based演化史]] — 从 DP 到 Genie/GameNGen
- [[03-流程/MARL全景综述]] — 从博弈论到 CICERO

---

## 📚 知识深度（三层架构）

### 01-原子/ — 基础概念

最小不可再分的知识点，被多个算法复用。

**数学基础**
- [[01-原子/Bellman方程]] — 动态规划的核心方程
- [[01-原子/TD误差]] — Bellman方程的采样形式
- [[01-原子/KL散度]] — Forward/Reverse KL 的直觉与数学

**策略优化**
- [[01-原子/策略梯度]] — 直接优化策略参数
- [[01-原子/优势函数]] — 降低策略梯度方差
- [[01-原子/GAE]] — 广义优势估计，平衡偏差-方差
- [[01-原子/Clip机制]] — 限制策略更新幅度
- [[01-原子/Trust-Region]] — KL散度硬约束

**探索与采样**
- [[01-原子/最大熵原理]] — 鼓励探索，防止过早收敛
- [[01-原子/重要性采样]] — 用行为策略的数据估计目标策略
- [[01-原子/重参数化技巧]] — 让随机采样操作可微分
- [[01-原子/经验回放]] — Off-policy的核心机制
- [[01-原子/目标网络]] — 稳定训练的关键技巧

**奖励建模**
- [[01-原子/Bradley-Terry模型]] — 偏好建模的理论基础
- [[01-原子/Reward-Model训练方法]] — 八大训练方法全景

### 02-模块/ — 完整算法

由多个原子组合而成的完整算法或方法族。

**Value-Based（价值学习）**
- [[02-模块/Value-Based/DQN]] — 深度 Q 网络
- [[02-模块/Value-Based/Q-Learning]] — 经典 Q 学习
- [[02-模块/Value-Based/CQL]] — 保守 Q 学习（Offline RL）

**Policy-Based（策略学习）**
- [[02-模块/Policy-Based/PPO]] — 近端策略优化
- [[02-模块/Policy-Based/TRPO]] — 信赖域策略优化
- [[02-模块/Policy-Based/REINFORCE]] — 经典策略梯度

**Actor-Critic（演员-评论家）**
- [[02-模块/Actor-Critic/SAC]] — 软演员-评论家
- [[02-模块/Actor-Critic/DDPG]] — 深度确定性策略梯度
- [[02-模块/Actor-Critic/TD3]] — 双延迟 DDPG

**奖励模型**
- [[02-模块/Reward-Model/DPO]] — 直接偏好优化
- [[02-模块/Reward-Model/PRM]] — 过程奖励模型
- [[02-模块/Reward-Model/Verifiable-Reward]] — 可验证奖励

**模型学习**
- [[02-模块/Model-Based/Dyna-Q]] — 动态规划 + Q 学习
- [[02-模块/Model-Based/MBPO]] — 基于模型策略优化

**模仿学习**
- [[02-模块/Imitation-Learning/IRL]] — 逆向强化学习
- [[02-模块/Imitation-Learning/知识蒸馏]] — 知识迁移

### 03-流程/ — 演化脉络

算法之间的历史演进关系和应用场景。

**演化综述**
- [[03-流程/Value-Based演化史]] — 18 个方法的完整演化链
- [[03-流程/Policy-AC演化史]] — 14 个方法 + 三条演化线
- [[03-流程/Model-Based演化史]] — 21 个方法 + 三条演化线
- [[03-流程/Offline-RL与RLHF演化史]] — 21+ 个方法 + 两大主线
- [[03-流程/MARL全景综述]] — 值分解、策略梯度、通信、自我博弈

**应用场景**
- [[03-流程/场景选择决策树]] — 如何根据问题选择合适的 RL 方法

---

## 🎯 应用领域

### 05-应用领域/Post-Training — LLM 后训练

用 RL 对齐大语言模型与人类偏好。

**核心方法**
- [[05-应用领域/Post-Training/RLHF]] — 人类反馈强化学习（SFT → RM → PPO）
- [[05-应用领域/Post-Training/DPO]] — 直接偏好优化（无需奖励模型）
- [[05-应用领域/Post-Training/GRPO]] — 群组相对策略优化（无需价值模型）
- [[05-应用领域/Post-Training/DAPO]] — 动态自适应偏好优化

**方法对比**
- [[05-应用领域/Post-Training/Post-Training方法全景]] — RLHF vs DPO vs GRPO vs DAPO

**前沿方向**
- Verifiable Reward（可验证奖励）
- Process Reward（过程奖励）
- Self-Rewarding（自我奖励）

### 05-应用领域/Multi-Agent — 多智能体系统

多个智能体协作或竞争的复杂系统。

**核心方法**
- [[05-应用领域/Multi-Agent/MARL概览]] — 多智能体强化学习全景
- [[05-应用领域/Multi-Agent/MADDPG]] — 多智能体 DDPG
- [[05-应用领域/Multi-Agent/QMIX]] — 单调价值函数分解

**关键概念**
- [[01-原子/CTDE]] — 集中训练分散执行
- [[01-原子/信用分配]] — 团队协作的奖励分配

**应用场景**
- 机器人协作
- 自动驾驶车队
- 游戏 AI（AlphaStar、OpenAI Five）

### 05-应用领域/Mid-Training — LLM 中期训练

在预训练和后训练之间，用 RL 提升模型能力。

**核心方法**
- [[05-应用领域/Mid-Training/Mid-Training概览]] — 中期训练全景
- [[05-应用领域/Mid-Training/SPIN]] — 自我博弈微调

**关键技术**
- [[01-原子/课程学习]] — 从易到难的训练策略
- Self-Play（自我博弈）
- Expert Iteration（专家迭代）

**前沿方向**
- DeepSeek-R1（推理 RL）
- OpenAI o1（Chain-of-Thought RL）

### 05-应用领域/娱乐应用 — 泛娱乐场景

RL 在娱乐、社交、创意内容领域的应用。

**虚拟角色陪伴**
- [[05-应用领域/娱乐应用/角色陪伴/Character-AI技术栈]] — 人设注入、情感对齐、长期记忆
- 产品：Character.AI、Replika、星野

**AI 图像生成**
- [[05-应用领域/娱乐应用/图像生成/RLHF-for-Image-Generation]] — Diffusion + RLHF
- 技术：HPS、ImageReward、DDPO

**游戏 AI**
- [[05-应用领域/娱乐应用/游戏与内容/游戏AI与RL]] — 竞技 AI、智能 NPC、动态难度
- 案例：AlphaStar、OpenAI Five、CICERO

---

## 🗂️ 分类维度（00-索引/）

按不同维度对 RL 算法进行分类：

**按优化对象**
- [[00-索引/按优化对象]] — Value-Based / Policy-Based / Actor-Critic

**按策略类型**
- [[00-索引/按策略类型]] — On-Policy / Off-Policy

**按动作空间**
- [[00-索引/按动作空间]] — Discrete / Continuous

**按数据来源**
- [[00-索引/按数据来源]] — Online / Offline

**按模型依赖**
- [[00-索引/按模型依赖]] — Model-Free / Model-Based

**多视角导航**
- [[00-索引/Policy视角]] — 从策略优化角度串联知识
- [[00-索引/Reward视角]] — 从奖励建模角度串联知识

---

## 🌐 在线版本

本知识库已部署为在线网站，支持全文搜索、知识图谱、暗色模式：

https://sherkevin.github.io/rl-notes/

---

## 📝 文档模板

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

### 2026-06-24 — 目录结构重构

**问题**：目录层次混乱，应用方向散落在多处

**解决方案**：采用双维度架构
- **维度一**：知识深度（01-原子 → 02-模块 → 03-流程）
- **维度二**：应用领域（05-应用领域/）

**具体变更**：
- 新增 `05-应用领域/` 目录
- 移动 `02-模块/Post-Training` → `05-应用领域/Post-Training`
- 移动 `02-模块/Multi-Agent` → `05-应用领域/Multi-Agent`
- 移动 `02-模块/Mid-Training` → `05-应用领域/Mid-Training`
- 移动 `04-娱乐应用` → `05-应用领域/娱乐应用`
- 重写 README，清晰区分两个维度
- 批量更新所有文档链接

### 2026-06-22 — 三层架构重构

- 创建三层架构：原子 → 模块 → 流程
- 拆分长文档为独立模块
- 创建多视角索引

---

## 🤝 贡献指南

1. **新增原子**：如果一个概念被 3 个以上模块使用，考虑提取为原子
2. **新增模块**：完整算法或方法族，引用相关原子
3. **新增流程**：演化路径或应用场景，串联相关模块
4. **新增应用**：特定领域的应用案例，引用相关模块和原子
5. **更新链接**：使用 `[[路径/文档名]]` 格式，保持双链完整

---

## 📚 参考资源

- 书籍：Sutton & Barto "Reinforcement Learning: An Introduction"
- 课程：David Silver's RL Course
- 代码：OpenAI Spinning Up
- 代码：CleanRL
