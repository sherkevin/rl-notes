---
tags:
  - #model-based
  - #model-based
  - #off-policy
  - #online-rl
  - #continuous
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[02-模块/Model-Based/Dyna-Q|Dyna-Q]]
- [[02-模块/Model-Based/MBPO|MBPO]]
- [[02-模块/Actor-Critic/SAC|SAC]]

### 分类维度
- [[00-索引/按模型依赖|Model-Based]]
- [[00-索引/按动作空间|Continuous]]
- [[00-索引/按数据来源|Online-RL]]

### 组合构成（用到的原子与模块）
- [[01-原子/TD误差|TD 误差]]（Q 值更新信号）
- [[01-原子/经验回放|经验回放]]（真实数据 buffer）
- [[01-原子/epsilon-greedy|探索策略]]（数据采集）
- [[02-模块/Actor-Critic/SAC|SAC]]（底层 Model-Free 策略优化器）

---

# PETS（Probabilistic Ensemble Trajectory Sampling）

## 1. 核心概述

**一句话：PETS 用 5 个神经网络组成"委员会"预测未来，委员会分歧越大说明模型越不确定，规划时就自动变得保守——这是对抗模型误差的核心武器。**

- **类型：** Model-Based RL（Dyna 族）
- **架构：** 概率集成动力学模型 + CEM 轨迹优化
- **提出者：** Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine, NeurIPS 2018 (arXiv:1805.12114)
- **论文标题：** "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"
- **地位：** Dyna 路线的**关键里程碑**——首次用深度学习 + 概率集成解决了 Dyna-Q 的两个致命问题（模型无法泛化、无误差量化）。

---

## 2. 核心创新：用"委员会分歧"量化不确定性

### 解决的问题

[[02-模块/Model-Based/Dyna-Q|Dyna-Q]] 用表格记忆做模型，无法泛化到新状态。如果用神经网络替代表格，容量大了但引入了**模型偏差（bias）**——神经网络可能在训练数据覆盖不到的区域做出离谱的预测，策略优化器会利用这些错误预测，学到在真实环境中完全无效的策略。

**PETS 的答案：** 训练多个独立的动力学模型（集成），它们的**预测分歧**天然反映了模型的不确定性：
- 所有模型预测一致 → 模型确定 → 可以信任预测
- 模型预测分歧大 → 模型不确定 → 规划时自动保守

### 两种不确定性的统一捕获

$$\text{总不确定性} = \underbrace{\text{认知不确定性}}_{\text{模型不知道（数据不够）}} + \underbrace{\text{偶然不确定性}}_{\text{环境本身有噪声}}$$

概率集成同时捕获这两种：
- **认知不确定性：** 不同模型因为训练数据/初始化不同而给出不同预测 → 集成分歧
- **偶然不确定性：** 每个模型自己预测的方差 $\Sigma_{\theta_k}$ → 环境噪声

---

## 3. 算法流程

### 整体架构

```
真实环境交互 → 存入 Replay Buffer → 训练 K 个概率动力学模型
                                        ↓
                          用集成模型做轨迹采样规划
                                        ↓
                              CEM 优化动作序列
                                        ↓
                              执行最优动作，收集新数据
                                        ↓
                                    重复循环
```

### 详细步骤

```
初始化: Replay Buffer D, K 个动力学模型 {f_θ_1, ..., f_θ_K}

Loop 每个 episode:
    // === 1. 规划阶段 ===
    对当前状态 s_t:
        用 CEM 优化 H 步动作序列 [a_t, a_{t+1}, ..., a_{t+H-1}]:
            重复 J 轮:
                从当前分布采样 N 条动作序列
                对每条序列，用集成模型做轨迹采样:
                    每一步随机选一个模型 k ~ Uniform(1..K)
                    s_{t+1} ~ N(μ_θ_k(s_t, a_t), Σ_θ_k(s_t, a_t))
                计算每条轨迹的累积奖励
                选 Top-M 条精英轨迹
                用精英轨迹更新 CEM 分布参数（均值、方差）
            返回 CEM 最终分布的均值作为最优动作序列
    执行 a_t, 观察 r_t, s_{t+1}

    // === 2. 模型学习阶段 ===
    将 (s_t, a_t, r_t, s_{t+1}) 存入 D
    从 D 中采样 batch，更新 K 个模型的参数
        每个模型独立训练，最大化高斯对数似然:
            L_k = -Σ log N(s_{t+1} | μ_θ_k(s_t, a_t), Σ_θ_k(s_t, a_t))
```

### 轨迹采样 (Trajectory Sampling) 的关键

这是 PETS 区别于普通集成方法的核心。规划时**每一步随机选一个模型**预测，而不是用集成均值：

- **用均值：** 不同模型的预测取平均 → 不确定性信息被平滑掉 → 规划器无法区分"模型确定"和"模型分歧但均值恰好相同"
- **随机选一个：** 集成分歧大的区域，不同采样轨迹会发散 → 平均奖励自然变低 → CEM 自动避开这些区域

---

## 4. 关键公式

### 概率动力学模型

每个模型 $k$ 预测下一步状态的均值和方差：

$$\hat{s}_{t+1} \sim \mathcal{N}(\mu_{\theta_k}(s_t, a_t), \Sigma_{\theta_k}(s_t, a_t))$$

### 集成统计量

集成的整体预测均值和方差：

$$\mu(s_t, a_t) = \frac{1}{K} \sum_{k=1}^{K} \mu_{\theta_k}(s_t, a_t)$$

$$\sigma^2(s_t, a_t) = \frac{1}{K} \sum_{k=1}^{K} \left[\sigma^2_{\theta_k}(s_t, a_t) + \mu^2_{\theta_k}(s_t, a_t)\right] - \mu^2(s_t, a_t)$$

**直觉：** 这就是"全方差公式"——总方差 = 各方差均值 + 各均值方差。第一项是偶然不确定性（每个模型自身的噪声），第二项是认知不确定性（模型间的分歧）。

### 模型训练损失（负对数似然）

$$\mathcal{L}_k = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{1}{2} \log |\Sigma_{\theta_k}| + \frac{1}{2} (s'_{i} - \mu_{\theta_k})^T \Sigma_{\theta_k}^{-1} (s'_{i} - \mu_{\theta_k}) \right]$$

### CEM（交叉熵方法）规划

每轮从 $\mathcal{N}(\mu_{cem}, \sigma^2_{cem})$ 采样 $N$ 条动作序列，选 Top-$M$ 精英，更新：

$$\mu_{cem} \leftarrow \frac{1}{M} \sum_{j=1}^{M} a^{(j)}_{elite}, \quad \sigma^2_{cem} \leftarrow \frac{1}{M} \sum_{j=1}^{M} (a^{(j)}_{elite} - \mu_{cem})^2$$

---

## 5. 真实例子与直觉理解

### 直觉：五个天气预报员

想象你要决定明天是否带伞，你问了 5 个天气预报员（5 个模型）：
- **情况 A：** 5 个人都说"80% 概率下雨" → 你很确定，果断带伞
- **情况 B：** 3 人说"一定下雨"，2 人说"一定晴天" → 分歧很大，不确定性高

PETS 的轨迹采样相当于：**每次从 5 人中随机抽一个来听他的建议，然后做模拟。** 如果 5 人意见一致，模拟结果收敛；如果分歧大，模拟结果发散，平均奖励低，CEM 自然避开这个方案。

### 实验结果（论文 Table 1）

在 HalfCheetah 等 MuJoCo 连续控制任务上，达到目标性能所需的**环境交互步数**：

| 方法 | 所需步数（相对比例） |
|------|:---:|
| PETS | **1x**（基线） |
| SAC (Model-Free SOTA) | ~8x |
| PPO | ~125x |

PETS 比 Model-Free 方法少 **1-2 个数量级**的环境交互。

---

## 6. 优缺点

### 优点
- **样本效率极高：** 比 [[02-模块/Actor-Critic/SAC|SAC]] 少 8 倍、比 PPO 少 125 倍样本
- **不确定性量化有效：** 集成方差防止策略利用模型错误（exploitation of model errors）
- **架构清晰：** 模型学习和策略优化解耦，各自可以独立改进
- **容易实现：** 核心代码不长，集成训练和 CEM 都是标准组件

### 缺点
- **计算成本高：** 需要训练 $K=5$ 个网络；CEM 规划每步需要评估数百条轨迹
- **只适用于低维连续控制：** 高维状态（如图像）效果不好
- **CEM 在高维动作空间中效率下降：** 动作维度越高，CEM 需要的采样数指数增长
- **模型和策略分开优化：** 没有端到端训练，模型可能在"对决策不重要"的维度上浪费容量
- **长 rollout 误差累积：** 虽然不确定性量化缓解了问题，但 25-40 步 rollout 后误差仍可能很大

---

## 7. 演化位置

```
Dyna-Q (1991) — tabular 模型，无误差量化
    ↓ 怎么让模型泛化？怎么量化不确定性？
PETS (2018) ← 你在这里！概率集成 NN + 轨迹采样
    ↓ 该多信任模型？rollout 该多长？
MBPO (2019) — 理论分析 + 极短 rollout
    ↓ 进一步...
TD-MPC2 (2024) — 隐式模型 + 跨域泛化
```

**在 [[03-流程/Model-Based演化史|Model-Based 演化史]] 中的位置：** Dyna → PETS → MBPO 主线的**中间关键节点**（详见 [[03-流程/Model-Based演化史|Model-Based 演化史 §7]]）。

### PETS 对后续工作的影响

| 影响 | 具体体现 |
|------|----------|
| 概率集成被广泛采用 | [[02-模块/Model-Based/MBPO|MBPO]] 直接沿用 PETS 的 5 网络集成 |
| 不确定性量化思想 | Dreamer 系列的 KL 散度正则化本质上也是不确定性约束 |
| CEM 规划 | 被 MPPI 等更高效的规划器逐步替代（如 TD-MPC2） |

---

## 8. 分类与相关

- **RL 分类：** Model-Based RL（Dyna 族）
- **数据来源：** [[00-索引/按数据来源|Online-RL]]
- **动作空间：** 连续（Continuous）
- **演化路线：** [[02-模块/Model-Based/Dyna-Q|Dyna-Q]] → PETS → [[02-模块/Model-Based/MBPO|MBPO]]（详见 [[03-流程/Model-Based演化史|Model-Based 演化史 §7]]）
- **前身：** [[02-模块/Model-Based/Dyna-Q|Dyna-Q]]（Dyna 架构的深度学习升级）
- **后继：** [[02-模块/Model-Based/MBPO|MBPO]]（理论化 + 短 rollout 改进）
- **对比：** World Models / Dreamer（潜空间想象路线，不同哲学）
