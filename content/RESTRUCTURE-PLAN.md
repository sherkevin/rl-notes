# RL 知识库三层架构重构计划

## 问题诊断

当前结构的问题：
1. **知识点重复**：DQN.md 详细写了经验回放、目标网络，这些概念在其他算法文件中重复出现
2. **层次不清**：原子概念（Bellman、KL散度）和模块（DQN、PPO）混在一起
3. **缺少复用**：基础构建块散落在各处，无法被多个算法引用
4. **视角单一**：主要是按算法类型组织，缺少多维视角（policy角度、reward角度等）

## 目标架构

采用 **原子-模块-流程** 三层架构，充分利用 Obsidian 双链：

```
03-流程/ (Flow)      ← 演化路径、应用场景、决策流程
   ↓ 组合多个模块
02-模块/ (Module)    ← 完整算法、方法族（DQN = 原子A + 原子B + 原子C）
   ↓ 组合多个原子
01-原子/ (Atomic)    ← 最小不可再分知识点（Bellman、TD误差、经验回放）

00-索引/ (Index)     ← 多维度入口（按优化对象、策略类型、动作空间等）
```

### 三层定义

**原子层 (01-原子/)**
- 最小不可再分的知识点
- 被多个模块复用
- 每个文件聚焦一个概念
- 示例：Bellman方程、TD误差、经验回放、目标网络、KL散度、策略梯度、优势函数、GAE

**模块层 (02-模块/)**
- 完整算法或方法族
- 由多个原子组合而成
- 通过双链引用原子，不重复内容
- 示例：DQN、PPO、SAC、GRPO、CQL

**流程层 (03-流程/)**
- 演化路径：模块之间的历史演进
- 应用场景：如何根据问题选择模块
- 决策流程：SFT → RM → PPO 等
- 示例：Value-Based演化史、RLHF完整流程、场景选择决策树

**索引层 (00-索引/)**
- 多维度入口页面
- 按不同视角组织知识
- 纯索引，不重复内容
- 示例：按优化对象、按策略类型、按动作空间、Policy视角、Reward Model视角

## 文件迁移计划

### 1. 创建新目录结构

```
rl/
├── 00-索引/
│   ├── README.md                    (主入口)
│   ├── 统一复习地图.md
│   ├── 按优化对象.md
│   ├── 按策略类型.md
│   ├── 按动作空间.md
│   ├── 按数据来源.md
│   ├── 按模型依赖.md
│   ├── Policy视角.md                (新增：从策略优化角度串联)
│   └── Reward视角.md                (新增：从奖励建模角度串联)
│
├── 01-原子/
│   ├── Bellman方程.md               (从基础概念拆分)
│   ├── TD误差.md                    (从基础概念拆分)
│   ├── 经验回放.md
│   ├── 目标网络.md
│   ├── epsilon-greedy.md
│   ├── 策略梯度.md                  (从Policy-Gradient提取)
│   ├── 优势函数.md                  (新增：从PPO提取)
│   ├── GAE.md                       (新增：从PPO提取)
│   ├── KL散度.md                    (从Forward-KL提取前半部分)
│   ├── 最大熵原理.md                (新增：从SAC提取)
│   ├── 重要性采样.md                (新增：从PPO提取)
│   ├── 重参数化技巧.md              (新增：从SAC提取)
│   ├── Trust-Region.md              (新增：从TRPO提取)
│   ├── Clip机制.md                  (新增：从PPO提取)
│   ├── Bradley-Terry模型.md         (新增：从Reward-Model提取)
│   └── IGM条件.md                   (新增：从MARL提取)
│
├── 02-模块/
│   ├── Value-Based/
│   │   ├── Q-Learning.md
│   │   ├── SARSA.md
│   │   ├── DQN.md                   (精简，引用原子)
│   │   ├── DDQN.md
│   │   ├── CQL.md
│   │   ├── BCQ.md
│   │   └── IQL.md
│   │
│   ├── Policy-Based/
│   │   ├── REINFORCE.md
│   │   ├── PPO.md                   (精简，引用原子)
│   │   ├── TRPO.md
│   │   ├── GRPO.md
│   │   └── DAPO.md
│   │
│   ├── Actor-Critic/
│   │   ├── A2C.md
│   │   ├── A3C.md
│   │   ├── DDPG.md
│   │   ├── TD3.md
│   │   ├── SAC.md
│   │   └── AWAC.md
│   │
│   ├── Model-Based/
│   │   ├── Dyna-Q.md
│   │   ├── PETS.md
│   │   └── MBPO.md
│   │
│   ├── Multi-Agent/
│   │   ├── VDN.md                   (新增)
│   │   ├── QMIX.md
│   │   ├── MADDPG.md
│   │   └── MAPPO.md
│   │
│   ├── Reward-Model/
│   │   ├── Pairwise-RM.md           (从Reward-Model拆分)
│   │   ├── Pointwise-RM.md
│   │   ├── PRM.md                   (Process Reward Model)
│   │   ├── ORM.md                   (Outcome Reward Model)
│   │   ├── LLM-as-Judge.md
│   │   ├── Generative-RM.md
│   │   ├── Verifiable-Reward.md
│   │   └── DPO.md                   (从Forward-KL提取后半部分)
│   │
│   └── Imitation-Learning/
│       ├── BC.md                    (新增)
│       ├── IRL.md
│       └── GAIL.md                  (新增)
│
└── 03-流程/
    ├── Value-Based演化史.md
    ├── Policy-AC演化史.md
    ├── Model-Based演化史.md
    ├── Offline-RL与RLHF演化史.md
    ├── MARL全景综述.md
    ├── RLHF完整流程.md              (新增：SFT→RM→PPO/DPO)
    └── 场景选择决策树.md            (新增)
```

### 2. 关键文件拆分策略

**Forward-KL与Reverse-KL.md (400行) → 拆分**
- `01-原子/KL散度.md`：KL散度定义、熵、Forward vs Reverse的数学和直觉
- `02-模块/Reward-Model/DPO.md`：DPO作为隐式Reward Model的应用
- `02-模块/Reward-Model/知识蒸馏.md`：Forward/Reverse KL在蒸馏中的应用

**Reward-Model训练方法.md (350行) → 拆分**
- `01-原子/Bradley-Terry模型.md`：Pairwise训练的理论基础
- `02-模块/Reward-Model/Pairwise-RM.md`：具体实现
- `02-模块/Reward-Model/Pointwise-RM.md`：具体实现
- `02-模块/Reward-Model/PRM.md`：Process Reward Model
- `02-模块/Reward-Model/ORM.md`：Outcome Reward Model
- `02-模块/Reward-Model/LLM-as-Judge.md`
- `02-模块/Reward-Model/Generative-RM.md`
- `02-模块/Reward-Model/Verifiable-Reward.md`

**DQN.md (270行) → 精简**
- 删除经验回放、目标网络的详细解释，改为引用原子：`详见 [[01-原子/经验回放]]`
- 保留DQN特有的内容：神经网络近似、Double DQN、Dueling DQN
- 添加组合说明：`DQN = [[01-原子/Bellman方程]] + [[01-原子/经验回放]] + [[01-原子/目标网络]] + 神经网络近似`

**PPO.md → 精简**
- 删除策略梯度、优势函数、GAE的详细推导，改为引用原子
- 保留PPO特有的内容：clip机制、ratio计算、训练流程
- 添加组合说明：`PPO = [[01-原子/策略梯度]] + [[01-原子/GAE]] + [[01-原子/Clip机制]] + Actor-Critic架构`

### 3. 新增索引页面

**00-索引/Policy视角.md**
从策略优化角度串联知识：
- 策略梯度基础：REINFORCE → 策略梯度定理
- 降低方差：Baseline → Advantage → GAE
- 稳定训练：Trust Region (TRPO) → Clip (PPO)
- 连续控制：DDPG → TD3 → SAC
- LLM对齐：PPO → GRPO → DAPO → DPO

**00-索引/Reward视角.md**
从奖励建模角度串联知识：
- 传统Reward：人工设计 → Reward Shaping
- 学习Reward：Pairwise RM → Pointwise RM → PRM/ORM
- 现代Reward：LLM-as-Judge → Generative RM → Verifiable Reward
- 隐式Reward：DPO → 策略即Reward

### 4. Tag系统设计

**层级tags**（用于分类）：
```yaml
# 优化对象
tags: [value-based, policy-based, actor-critic]

# 策略类型
tags: [on-policy, off-policy]

# 动作空间
tags: [discrete, continuous]

# 数据来源
tags: [online, offline]

# 模型依赖
tags: [model-free, model-based]
```

**领域tags**（用于主题）：
```yaml
tags: [llm-alignment, multi-agent, model-based, imitation-learning]
```

**状态tags**（用于文档管理）：
```yaml
tags: [draft, stub, complete]
```

### 5. 文档模板

**原子文档模板** (01-原子/)
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

**模块文档模板** (02-模块/)
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

**流程文档模板** (03-流程/)
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

## 实施步骤

### Phase 1: 创建新结构
1. 创建 `01-原子/`、`02-模块/`、`03-流程/`、`00-索引/` 目录
2. 移动现有文件到新位置（不改内容）

### Phase 2: 拆分长文档
1. 拆分 `Forward-KL与Reverse-KL.md` → KL散度 + DPO + 知识蒸馏
2. 拆分 `Reward-Model训练方法.md` → 8个独立RM模块
3. 从现有算法文件中提取原子概念

### Phase 3: 精简模块文档
1. DQN.md：删除重复内容，添加原子引用
2. PPO.md：删除重复内容，添加原子引用
3. 其他算法文件：同样处理

### Phase 4: 创建新索引
1. 编写 `Policy视角.md`
2. 编写 `Reward视角.md`
3. 更新 `README.md` 指向新结构

### Phase 5: 更新链接
1. 修复所有内部 `[[wikilink]]`
2. 添加新文档之间的交叉引用
3. 验证无断链

### Phase 6: 同步Quartz
1. rsync到quartz-rl
2. build测试
3. git push部署

## 预期收益

1. **零重复**：每个知识点只写一次，通过双链复用
2. **多层次理解**：
   - 想学基础？→ 01-原子/
   - 想学算法？→ 02-模块/
   - 想看演化？→ 03-流程/
   - 想找入口？→ 00-索引/
3. **多视角导航**：
   - Policy角度：从策略梯度到DPO
   - Reward角度：从Pairwise RM到Verifiable Reward
   - 应用角度：从问题到算法选择
4. **易于扩展**：新算法只需写模块文档，引用现有原子即可

## 风险与缓解

**风险1：链接大量断裂**
- 缓解：Phase 5专门处理，使用脚本批量更新

**风险2：拆分导致内容碎片化**
- 缓解：每个文档保持自包含（一句话定义+核心内容），双链只是补充而非必需

**风险3：用户不适应新结构**
- 缓解：保留旧README，添加新旧结构对照表；00-索引/提供多维度入口

## 时间估计

- Phase 1-2: 30分钟（创建结构、拆分文档）
- Phase 3-4: 40分钟（精简模块、创建索引）
- Phase 5-6: 20分钟（更新链接、同步部署）
- 总计：约90分钟
