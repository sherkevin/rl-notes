---
tags:
  - #model-based
  - #value-based
  - #off-policy
  - #online-rl
  - #discrete
  - #tabular
---

## [Link] 知识图谱链接

### 相关算法
- [[02-模块/Value-Based/Q-Learning|Q-Learning]]
- [[02-模块/Model-Based/PETS|PETS]]
- [[02-模块/Model-Based/MBPO|MBPO]]

### 分类维度
- [[00-索引/按模型依赖|Model-Based]]
- [[00-索引/按动作空间|Discrete]]
- [[00-索引/按数据来源|Online-RL]]

### 组合构成（用到的原子）
- [[01-原子/Q-Learning|Q-Learning]]（直接学习部分）
- [[01-原子/Bellman方程|Bellman 方程]]（规划部分的理论基础）
- [[01-原子/epsilon-greedy|epsilon-greedy]]（探索策略）
- [[01-原子/TD误差|TD 误差]]（Q 值更新信号）

---

# Dyna-Q（动态规划 + Q-Learning）

## 1. 核心概述

**一句话：Dyna-Q 是 Model-Based RL 的鼻祖——让 Agent 一边从真实经验学模型，一边用模型"做梦"生成额外经验来加速学习。**

- **类型：** Model-Based + Model-Free 混合
- **架构：** Q-Learning + Tabular Dynamics Model
- **提出者：** Richard Sutton, 1991, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting"
- **地位：** MBRL 从理论走向实践的**第一步**，后续 [[02-模块/Model-Based/PETS|PETS]](2018) 和 [[02-模块/Model-Based/MBPO|MBPO]](2019) 都是这条路线的直接继承者。

---

## 2. 核心创新：为什么要"做梦"？

### 解决的问题

纯 Model-Free 的 [[02-模块/Value-Based/Q-Learning|Q-Learning]] 每跟环境交互一步，只能做一次 Q 值更新。如果环境交互很昂贵（机器人、真实物理实验），这种"学一步用一步"的方式太慢。

**Dyna-Q 的答案：** 把每次真实交互的经验"物尽其用"——

1. 直接用真实经验更新 Q 值（Model-Free 部分）
2. 把真实经验存入一个**表格模型**（Model Learning）
3. 从模型中随机回放历史状态-动作对，生成**模拟经验**，再用这些模拟经验更新 Q 值（Planning 部分）

**直觉理解：** 就像你白天做了件事（真实交互），晚上睡觉时大脑会把白天的经验反复"重放"（规划），从而加速记忆巩固。Dyna-Q 的 planning steps 就是这个"做梦"过程。

---

## 3. 算法流程

### Dyna-Q 主循环

```
初始化 Q(s,a) 和 Model(s,a) 为空

Loop 每个 episode:
    初始化状态 S
    Loop 每个时间步:
        A ← epsilon-greedy(Q, S)           // 探索策略选动作
        执行 A, 观察 R, S'                  // 真实环境交互
        
        // === 1. 直接学习 (Direct RL) ===
        用 (S, A, R, S') 做一次 Q-Learning 更新
        
        // === 2. 模型学习 (Model Learning) ===
        Model(S, A) ← (R, S')              // 记录到表格模型
        
        // === 3. 规划 (Planning) ===
        重复 n 次:
            S_prev ← 随机选一个模型中见过的状态
            A_prev ← 随机选一个在 S_prev 下做过的动作
            (R_pred, S'_pred) ← Model(S_prev, A_prev)
            用 (S_prev, A_prev, R_pred, S'_pred) 做一次 Q-Learning 更新
        
        S ← S'
```

### 三个组件的职责

| 组件 | 做什么 | 类比 |
|------|--------|------|
| **Direct RL** | 真实经验直接更新 Q 值 | 白天做事直接学到东西 |
| **Model Learning** | 把真实经验存入表格 | 白天经历的事情被记录 |
| **Planning** | 从模型回放 n 步模拟经验 | 晚上做梦重放白天经历 |

---

## 4. 关键公式

### Q-Learning 更新（Direct RL 和 Planning 都用同一个公式）

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right]$$

其中：
- $\alpha$：学习率
- $\gamma$：折扣因子
- $R + \gamma \max_{a'} Q(S', a')$ 是 [[01-原子/TD误差|TD 目标]]

### 模型学习（确定性环境的表格模型）

$$Model(s, a) \leftarrow (r, s')$$

对于确定性环境，直接记忆即可。如果是随机环境，可以记录转移概率或采样。

### 规划步数的效果

设 $n$ 为每步真实交互后的 planning 步数：
- $n = 0$：退化为纯 [[02-模块/Value-Based/Q-Learning|Q-Learning]]
- $n$ 越大：Q 值收敛越快（每步真实交互被"复用"更多次），但计算成本增加
- **关键权衡：** $n$ 太大时，如果模型有误差，Q 值会被错误的模拟经验带偏

---

## 5. 真实例子：迷宫任务

**场景：** Agent 在一个 $6 \times 9$ 的迷宫中找出口。

**实验结果（Sutton & Barto 教材第 8 章）：**

| Planning 步数 $n$ | 收敛到最优策略所需的 episode 数 |
|:---:|:---:|
| 0（纯 Q-Learning） | ~50 |
| 5 | ~10 |
| 50 | ~5 |

**直觉：** $n=0$ 的 Agent 每走一步只学一次，信息传播极慢（只有走过终点附近才知道终点好）。$n=50$ 的 Agent 每走一步就在"梦里"反复演练，Q 值的更新像涟漪一样快速扩散到整个状态空间。

---

## 6. 优缺点

### 优点
- **样本效率高：** 每步真实交互被复用 $n$ 次，大幅减少环境交互需求
- **架构简洁：** 三个组件（学、记、梦）清晰分离又协同工作
- **首次证明可行性：** "学模型 + 用模型"这个范式在 Dyna-Q 之前只是理论构想
- **Anytime 性质：** 随时可以停止规划，使用当前 Q 值行动

### 缺点
- **Tabular 模型无泛化：** 直接记忆 $(s,a) \to (r,s')$，没见过的新状态无法预测
- **无模型误差处理：** 如果模型记错了，Q 值会被错误经验带偏（没有任何纠错机制）
- **只适用离散小空间：** 状态和动作都必须是离散有限的
- **规划效率低：** 随机采样 $(s,a)$ 对做规划，没有优先级——可能反复回放不重要的经验

---

## 7. 演化位置

```
Bellman 动态规划 (1950s)
    ↓ 模型未知怎么办？
Dyna-Q (1991) ← 你在这里！首次从经验学模型 + 用模型规划
    ↓ 模型怎么泛化？误差怎么处理？
PETS (2018) → 用概率集成 NN 替代表格模型，用集成方差量化不确定性
    ↓ 该多信任模型？rollout 该多长？
MBPO (2019) → 理论分析得出"短 rollout"最优，工程上极简洁
    ↓ 进一步...
TD-MPC2 (2024) → 隐式模型 + 跨域泛化
```

**在 [[03-流程/Model-Based演化史|Model-Based 演化史]] 中的位置：** Dyna → PETS → MBPO 主线（截断误差传播路线）的**起点**。核心思想"从真实数据出发做短 rollout"贯穿整条路线。

### Dyna-Q 没解决的问题催生了什么？

| 问题 | 后续解法 |
|------|----------|
| 表格模型无法泛化 | [[02-模块/Model-Based/PETS|PETS]] 用神经网络动力学模型 |
| 没有模型误差量化 | [[02-模块/Model-Based/PETS|PETS]] 用集成方差量化不确定性 |
| rollout 长度怎么选 | [[02-模块/Model-Based/MBPO|MBPO]] 理论分析 → 短 rollout 最优 |
| 规划效率低（随机采样） | Prioritized Sweeping（TD 误差大的优先回放） |

---

## 8. 分类与相关

- **RL 分类：** Model-Based RL（Dyna 族）
- **数据来源：** [[00-索引/按数据来源|Online-RL]]
- **动作空间：** 离散（Discrete）
- **演化路线：** Dyna → [[02-模块/Model-Based/PETS|PETS]] → [[02-模块/Model-Based/MBPO|MBPO]]（详见 [[03-流程/Model-Based演化史|Model-Based 演化史 §4]]）
- **前身：** [[01-原子/Bellman方程|Bellman 动态规划]]（理论基石）
- **后继：** [[02-模块/Model-Based/PETS|PETS]]（概率集成版 Dyna）、[[02-模块/Model-Based/MBPO|MBPO]]（理论化 Dyna）
- **变体：** Dyna-2（结合 TD learning）、Prioritized Sweeping（优先级规划）
