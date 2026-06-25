---
tags:
  - #model-based
  - #actor-critic
  - #off-policy
  - #online-rl
  - #continuous
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[02-模块/Model-Based/Dyna-Q|Dyna-Q]]
- [[02-模块/Model-Based/PETS|PETS]]
- [[02-模块/Actor-Critic/SAC|SAC]]
- [[02-模块/Policy-Based/PPO|PPO]]

### 分类维度
- [[00-索引/按模型依赖|Model-Based]]
- [[00-索引/按动作空间|Continuous]]
- [[00-索引/按数据来源|Online-RL]]

### 组合构成（用到的原子与模块）
- [[01-原子/策略梯度|策略梯度]]（SAC 的 actor 更新）
- [[01-原子/重参数化技巧|重参数化技巧]]（SAC 动作采样）
- [[01-原子/经验回放|经验回放]]（真实 + 模型混合 buffer）
- [[01-原子/KL散度|KL 散度]]（理论分析中的策略约束）
- [[01-原子/TD误差|TD 误差]]（critic 更新信号）
- [[02-模块/Actor-Critic/SAC|SAC]]（底层 Model-Free 策略优化器）
- [[02-模块/Model-Based/PETS|PETS]] 的概率集成动力学模型（直接沿用）

---

# MBPO（Model-Based Policy Optimization）

## 1. 核心概述

**一句话：MBPO 用数学证明了"模型不可信太远"，然后给出了最简洁的工程方案——从真实数据出发只做 1 步 rollout，混合 5% 模型数据训练 SAC，就接近了 PETS 的样本效率且最终性能更好。**

- **类型：** Model-Based RL（Dyna 族）
- **架构：** 概率集成动力学模型 + 短 rollout + SAC
- **提出者：** Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine, NeurIPS 2019 (arXiv:1906.08253)
- **论文标题：** "When to Trust Your Model: Model-Based Policy Optimization"
- **地位：** Dyna 路线的**成熟之作**——理论回答了"该多信任模型"这个根本问题，工程上简洁有效，成为后续 MBRL 工作的标准 baseline。

---

## 2. 核心创新：理论指导的"短 rollout"

### 解决的问题

[[02-模块/Model-Based/PETS|PETS]] 虽然样本效率高，但有两个问题：
1. **计算成本高：** CEM 在线规划需要每步评估数百条轨迹
2. **长 rollout 误差累积：** 25-40 步 rollout 后，即使有不确定性量化，模型误差仍会累积

**核心问题：** 到底该多信任模型？rollout 该做多长？

### MBPO 的答案

**理论分析：** 推导了策略改进的下界，证明 rollout 越长，模型误差的惩罚越大。

**工程方案：** 用**极短 rollout**（通常 1 步，最多几步）从真实数据 buffer 中分支出来，混合真实数据和模型数据训练 [[02-模块/Actor-Critic/SAC|SAC]]。

### 直觉理解：为什么短 rollout 最优？

想象你在学开车：
- **纯 Model-Free（$n=0$）：** 只在真实路上练，每次只能学一小段
- **长 rollout 想象（$n=100$）：** 在脑中想象开 100 公里，但想象力有限，越往后越偏离现实，可能练出"幻觉车技"
- **短 rollout 想象（$n=1$）：** 每次真实开一段后，在脑中只"预演"下一步会怎样，然后马上回真实路面验证。这样既加速了学习，又不会被想象带偏

---

## 3. 算法流程

### 整体架构

```
真实环境交互 → 真实 Replay Buffer D_real
                    ↓
            训练 K 个集成动力学模型（同 PETS）
                    ↓
        从 D_real 中采样真实状态 s_t
        用集成模型做 k 步 rollout（k 很小，通常 1）
        生成模型数据 → 模型 Buffer D_model
                    ↓
        混合采样：95% D_real + 5% D_model
                    ↓
            用 SAC 更新策略（Actor + Critic）
                    ↓
                重复循环
```

### 详细步骤

```
初始化: 真实 Buffer D_real, 模型 Buffer D_model, 
        K 个动力学模型 {f_θ_1, ..., f_θ_K}, SAC 策略 π

Loop 每个 epoch:
    // === 1. 数据采集 ===
    用当前策略 π 在真实环境中交互 N 步
    存入 D_real
    
    // === 2. 模型训练 ===
    从 D_real 采样 batch，更新 K 个动力学模型
    （同 PETS 的负对数似然损失）
    
    // === 3. 模型 rollout（短！）===
    重复 M 次:
        从 D_real 随机采样一个状态 s_t
        用 π 采样动作 a_t ~ π(·|s_t)
        随机选一个集成模型 k
        预测下一步: s_{t+1}, r_{t+1} ~ f_θ_k(s_t, a_t)
        将 (s_t, a_t, r_{t+1}, s_{t+1}) 存入 D_model
        // 如果 rollout 长度 k > 1，继续从 s_{t+1} 往前推
    
    // === 4. 策略更新（SAC）===
    重复 G 次梯度更新:
        以混合比例 f 采样 batch:
            f = |D_model| / (|D_real| + |D_model|) ≈ 0.05
            即 95% 来自 D_real, 5% 来自 D_model
        用 SAC 更新 Actor 和 Critic
```

### Rollout 长度 Schedule

训练过程中，随着模型越来越好，rollout 长度可以逐渐增加：

| 训练阶段 | Rollout 长度 $k$ | 原因 |
|:---:|:---:|:---|
| 早期 | 1 | 模型还不好，只做 1 步预测误差可控 |
| 中期 | 3 | 模型改善了，可以稍微多信一点 |
| 后期 | 5 | 模型比较准了，但仍需截断 |

---

## 4. 关键公式

### 策略改进下界（核心理论贡献）

$$\eta(\pi_{new}) \geq \eta(\pi_{old}) - \frac{2\gamma \epsilon_{model}}{(1-\gamma)^2} - \frac{2\gamma r_{max} \epsilon_{\pi}}{(1-\gamma)^2}$$

其中：
- $\eta(\pi)$：策略 $\pi$ 的真实期望回报
- $\epsilon_{model}$：模型误差（单步预测的最大偏差）
- $\epsilon_{\pi}$：新旧策略之间的 [[01-原子/KL散度|KL 散度]]偏移
- $\gamma$：折扣因子
- $r_{max}$：最大奖励

**核心洞察：** 两项惩罚都跟 $\frac{1}{(1-\gamma)^2}$ 成正比——这恰恰是 rollout 长度的函数。Rollout 越长，$\gamma$ 越接近 1，惩罚项爆炸式增长。因此**短 rollout 在理论上就是更安全的选择。**

### 混合比例

$$f = \frac{\text{model rollout 步数}}{\text{总梯度更新步数}} \approx 0.05$$

即模型数据占总训练数据的 ~5%。这个比例看似很小，但因为模型 rollout 几乎零成本（不需要真实环境交互），实际加速效果显著。

### 动力学模型（沿用 PETS 的概率集成）

$$\hat{s}_{t+1} \sim \mathcal{N}(\mu_{\theta_k}(s_t, a_t), \Sigma_{\theta_k}(s_t, a_t))$$

$K=5$ 个独立训练的模型，rollout 时每步随机选一个。

---

## 5. 真实例子与直觉理解

### MBPO vs PETS 的本质区别

| | PETS | MBPO |
|---|---|---|
| **规划方式** | 在线 CEM（每步都优化动作序列） | 离策略 SAC（提前训练好策略网络） |
| **推理成本** | 高（每步数百条轨迹评估） | 低（策略网络一次前向） |
| **rollout 长度** | 25-40 步 | 1-5 步 |
| **模型用法** | 在线规划（planning） | 数据增强（data augmentation） |

**直觉：** PETS 像每次决策前都"深度思考"（CEM 规划），MBPO 像"平时多练习"（SAC 离线训练）然后决策时快速反应。MBPO 的策略网络把模型的知识"编译"进了网络权重，推理时不再需要模型。

### 实验结果（论文 Figure 4 & 5）

在 Ant、HalfCheetah、Hopper、Walker2d 四个 MuJoCo 任务上：

| 方法 | 样本效率 | 最终性能 | 计算成本 |
|------|:---:|:---:|:---:|
| MBPO | 高（接近 PETS） | **最好** | 中等 |
| PETS | 高 | 次好 | 高 |
| SAC | 低 | 好 | 低 |
| PPO | 很低 | 一般 | 低 |

MBPO 的独特优势：**样本效率接近 PETS，但最终性能超过 PETS。** 原因是短 rollout 避免了长 rollout 的误差累积，策略不会被模型错误误导。

---

## 6. 优缺点

### 优点
- **理论上有策略改进保证：** 基于策略改进下界，不是纯启发式
- **样本效率接近 PETS：** 模型数据虽然只占 5%，但几乎零成本
- **最终性能更好：** 短 rollout 避免了模型误差累积，不 exploit 模型错误
- **计算成本比 PETS 低：** 不需要在线 CEM 规划，推理时只需策略网络前向
- **与 Model-Free 算法无缝结合：** 本质上是在 SAC 外面包了一层模型数据增强
- **实现简洁：** 核心改动就是"训练模型 + 短 rollout + 混合 buffer"

### 缺点
- **需要大量真实数据：** 早期模型不好时，rollout 生成的数据质量差，需要足够真实数据打底
- **Rollout 长度需要 schedule：** 手动设定训练各阶段的 rollout 长度
- **本质是数据增强：** 没有充分利用模型的规划能力（不像 PETS 那样在线规划）
- **模型误差处理仍粗糙：** 靠"短"截断误差，而非主动建模或纠正误差
- **混合比例 $f$ 需要调优：** 不同任务的最优混合比例不同

---

## 7. 演化位置

```
Dyna-Q (1991) — tabular 模型 + Q-learning
    ↓
PETS (2018) — 概率集成 NN + CEM 在线规划
    ↓ 该多信任模型？rollout 该多长？
MBPO (2019) ← 你在这里！理论分析 + 短 rollout + SAC
    ↓ 能否在潜空间想象，避免状态空间的误差累积？
Dreamer v1 (2020) — RSSM 潜空间想象 + Actor-Critic
    ↓
TD-MPC2 (2024) — 隐式模型 + 跨域泛化
```

**在 [[03-流程/Model-Based演化史|Model-Based 演化史]] 中的位置：** Dyna → PETS → MBPO 主线的**成熟之作**（详见 [[03-流程/Model-Based演化史|Model-Based 演化史 §8]]）。

### MBPO 的核心贡献总结

1. **理论回答了"该多信任模型"：** rollout 长度与模型误差的 trade-off 有了数学表达
2. **工程上极简洁：** 短 rollout + SAC 混合训练，容易实现和调优
3. **成为标准 baseline：** 后续 MBRL 工作几乎都要跟 MBPO 对比

### MBPO 的"模型信任度"思想的影响

| 后续工作 | 如何继承 |
|----------|----------|
| [[03-流程/Model-Based演化史#9-dreamer-v1--hafner-et-al-2020|Dreamer v1]] | 潜空间 rollout 也遵循"不宜太长"的原则 |
| MOPO (2020, Offline MBRL) | 在 MBPO 基础上加了保守惩罚项处理 OOD |
| TD-MPC2 (2024) | 用 TD 学习替代显式 rollout，进一步降低模型依赖 |

---

## 8. 分类与相关

- **RL 分类：** Model-Based RL（Dyna 族）
- **数据来源：** [[00-索引/按数据来源|Online-RL]]
- **动作空间：** 连续（Continuous）
- **演化路线：** [[02-模块/Model-Based/Dyna-Q|Dyna-Q]] → [[02-模块/Model-Based/PETS|PETS]] → MBPO（详见 [[03-流程/Model-Based演化史|Model-Based 演化史 §8]]）
- **前身：** [[02-模块/Model-Based/PETS|PETS]]（概率集成 + 在线规划）
- **底层优化器：** [[02-模块/Actor-Critic/SAC|SAC]]（Model-Free 策略优化）
- **后继：** MOPO（Offline 版本）、TD-MPC2（隐式模型路线）
- **对比：** PETS（同用集成模型，但 PETS 做在线规划，MBPO 做离策略训练）
