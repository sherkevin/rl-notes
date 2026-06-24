# 多智能体强化学习 (MARL) 全景综述

> **作者**:自动化调研 | **日期**:2026-06-21
> **范围**:从博弈论基础到 2024-2026 前沿(LLM-based agents)

---

## 目录

- [Part A: MARL 基础](#part-a-marl-基础)
- [Part B: 合作型值分解方法](#part-b-合作型值分解方法)
- [Part C: [[01-原子/策略梯度|策略梯度]]型 MARL](#part-c-[[01-原子/策略梯度|策略梯度]]型-marl)
- [Part D: 通信与涌现行为](#part-d-通信与涌现行为)
- [Part E: 高级主题与里程碑系统](#part-e-高级主题与里程碑系统)
- [Part F: 2024-2026 前沿](#part-f-2024-2026-前沿)

---

## Part A: MARL 基础

### A.1 为什么多智能体从根本上不同

单智能体 RL 的理论基石是马尔可夫决策过程(MDP),其核心假设是**环境转移概率是稳定的**。一旦引入多个学习智能体,这个假设就彻底崩塌。

**非平稳性 (Non-stationarity)**

在单智能体环境中,状态转移函数 P(s'|s,a) 是固定的。在多智能体环境中,对于智能体 i,它观察到的"环境"包含了其他所有智能体,而其他智能体的策略在不断更新。这意味着:

- 从智能体 i 的视角,状态转移变成了 P(s'|s, a_i, a_{-i}),其中 a_{-i} 是其他智能体的联合动作
- 其他智能体在不断学习,所以 a_{-i} 的分布在持续变化
- 智能体 i 面对的是一个**非平稳环境** -- [[01-原子/经验回放|经验回放]](replay buffer)中的旧数据可能完全失效

这是 MARL 的核心难题,也是为什么简单地把单智能体算法直接搬过来([[02-模块/Value-Based/IQL|IQL]])往往效果不好。

**部分可观测性 (Partial Observability)**

每个智能体只能看到自己的局部观测 o_i,而非完整状态 s。这把问题从 MDP 推向了 **Dec-POMDP(分散式部分可观测马尔可夫决策过程)**:

- 每个智能体维护自己的策略 pi_i(a_i | o_i)
- 最优联合策略需要考虑所有智能体的局部观测历史
- Dec-POMDP 在计算复杂度上是 NEXP-complete,远比 MDP 困难

**信用分配问题 (Credit Assignment)**

多个智能体共同行动后,收到一个团队奖励 r。问题是:谁的贡献大,谁在搭便车?

举例:在一个 3v3 的战斗中,团队赢了获得 +1 奖励。但实际可能是 1 号智能体做了关键操作,而 2 号和 3 号只是围观。如果没有好的信用分配机制,2 号和 3 号会"学会"继续围观 -- 这就是**搭便车问题 (free-rider problem)**。

信用分配的质量直接决定了合作型 MARL 算法的上限。不同的算法族本质上是在用不同方式回答这个问题。

### A.2 博弈论基础

MARL 的数学根基是博弈论。理解几个核心概念对于理解算法设计至关重要。

**Nash 均衡 (Nash Equilibrium)**

一组策略组合 (pi_1*, pi_2*, ..., pi_n*) 是 Nash 均衡,当且仅当没有任何一个智能体可以通过单方面改变自己的策略来提高收益:

> 对所有 i 和所有可选策略 pi_i: J_i(pi_i*, pi_{-i}*) >= J_i(pi_i, pi_{-i}*)

Nash 均衡不一定是全局最优的 -- 这就是为什么 MARL 中收敛到 Nash 均衡不一定意味着"好"的结果。

**Pareto 最优 (Pareto Optimality)**

一个结果 s 是 Pareto 最优的,当且仅当不存在另一个结果 s' 使得所有智能体的收益都不下降、至少有一个智能体的收益严格提高。

Nash 均衡 vs Pareto 最优的冲突在**社会困境**中尤为突出。

**社会困境 (Social Dilemma)**

经典的囚徒困境揭示了个体理性与集体理性的冲突:

| | 合作 (C) | 背叛 (D) |
|---|---|---|
| **合作 (C)** | (3, 3) | (0, 5) |
| **背叛 (D)** | (5, 0) | (1, 1) |

(D, D) 是唯一的 Nash 均衡,但 (C, C) 才是 Pareto 最优。在 MARL 中,如何让智能体学会合作而非陷入背叛均衡,是核心挑战之一。

**零和博弈 vs 一般和博弈**

- **零和博弈**:所有智能体的收益之和恒为 0(如棋类),竞争关系纯粹
- **一般和博弈**:收益之和可变(如贸易谈判),可能同时存在合作与竞争

### A.3 合作 / 竞争 / 混合设定

**合作型 (Cooperative)**

所有智能体共享同一个奖励函数:r_1 = r_2 = ... = r_n = r。目标是最大化团队总收益。

典型场景:星际争霸微操(SMAC)、足球配合、灾难救援

这是 MARL 研究最深入的设定,因为:
- 可以明确定义团队目标
- 信用分配问题可以用数学方法精确处理
- 值分解方法在此设定下有良好的理论基础

**竞争型 (Competitive)**

智能体利益对立,一方所得即另一方所失。典型为零和或常和博弈。

典型场景:棋类、扑克、格斗游戏

核心方法:自博弈(self-play),对抗训练,虚拟对手建模

**混合型 (Mixed / General-sum)**

智能体利益部分一致、部分冲突。这是最现实也最难的设定。

典型场景:交通流、拍卖机制、谈判

挑战:需要在合作与竞争之间找到平衡,可能涉及讨价还价、欺骗、联盟形成等复杂策略。

### A.4 集中式 vs 分散式

MARL 的训练和执行有两个维度,产生了四种组合:

| | **集中式执行** | **分散式执行** |
|---|---|---|
| **集中式训练** | 完全集中式(不实用) | **CTDE(主流范式)** |
| **分散式训练** | 理论上的可能性 | **完全分散式([[02-模块/Value-Based/IQL|IQL]])** |

**集中式训练 + 集中式执行**:要求执行时也能获取所有智能体的信息。在大多数实际场景中不可行(通信延迟、带宽限制、隐私要求)。

**分散式训练 + 分散式执行**:每个智能体独立学习。简单但面临非平稳性问题,难以学会协调。

**CTDE**:训练时利用全局信息,执行时只用局部观测。这是目前 MARL 的主导范式。

### A.5 CTDE: 集中式训练 + 分散式执行

CTDE 是目前 MARL 领域最重要的范式,几乎所有主流算法([[05-应用领域/Multi-Agent/QMIX|VDN]]、[[05-应用领域/Multi-Agent/QMIX|QMIX]]、[[05-应用领域/Multi-Agent/MADDPG|MADDPG]]、[[05-应用领域/Multi-Agent/MAPPO|MAPPO]])都在此框架下运作。

**核心思想**:

在训练阶段,我们可以访问所有智能体的观测、动作和全局状态 s。这让我们能够:
- 学习一个全局价值函数 Q_tot(s, a_1, ..., a_n)
- 更准确地评估每个智能体的贡献
- 利用全局信息进行更有效的信用分配

在执行阶段,每个智能体只根据本地观测 o_i 做决策:
- a_i = argmax_a Q_i(o_i, a)
- 不需要通信(或只需有限通信)

**为什么 CTDE 有效?**

1. **训练和执行的解耦**:训练时可以"作弊"使用全局信息,但通过结构性约束确保执行时不需要这些信息
2. **非平稳性缓解**:训练阶段的全局信息提供了更稳定的学习信号
3. **信用分配**:全局值函数可以精确分解,识别每个智能体的贡献
4. **实际可行性**:模拟环境/实验室环境中全局信息通常可获取

**关键约束 -- IGM(Individual-Global-Max)**:

为了确保集中式训练的结果可以分散执行,需要满足:

> argmax_{a_1,...,a_n} Q_tot(s, a_1, ..., a_n) = (argmax_a Q_1(o_1, a), ..., argmax_a Q_n(o_n, a))

即全局最优联合动作 = 每个智能体各自最优动作的组合。[[05-应用领域/Multi-Agent/QMIX|VDN]] 和 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 都是通过不同的方式保证这个性质。

---

## Part B: 合作型值分解方法

> **演化主线**: [[02-模块/Value-Based/IQL|IQL]] -> [[05-应用领域/Multi-Agent/QMIX|VDN]] -> [[05-应用领域/Multi-Agent/QMIX|QMIX]] -> Weighted [[05-应用领域/Multi-Agent/QMIX|QMIX]] -> QPLEX -> QATT
> **核心问题**: 如何把全局 Q 值分解为各智能体的局部 Q 值,同时保证 IGM 性质

### B.1 [[02-模块/Value-Based/IQL|IQL]] (Independent [[02-模块/Value-Based/Q-Learning|Q-Learning]])

**来源与动机**:最朴素的基线方法。把多智能体问题当作多个独立的单智能体问题。每个智能体用自己的 Q-learning 独立学习(Tan, 1993)。

**核心创新**:无创新,就是简单地将 Q-learning 应用到每个智能体。

**关键公式**:

每个智能体 i 独立维护 Q_i(o_i, a_i),更新规则:

Q_i(o_i, a_i) <- Q_i(o_i, a_i) + alpha * [r_i + gamma * max_{a'} Q_i(o_i', a') - Q_i(o_i, a_i)]

**优缺点**:
- 优点:实现极简,无需通信,天然分散式
- 缺点:完全忽视非平稳性,无法信用分配,在复杂协作任务中表现差
- 在某些简单环境中仍然有竞争力的表现(令人惊讶)

**演化位置**:值分解方法的"零号基线",所有后续方法都以"比 IQL 好多少"来证明自己的价值。

### B.2 VDN (Value Decomposition Networks)

**来源与动机**:Sunehag et al., 2017, DeepMind。首次提出值分解框架,解决 IQL 无法利用团队奖励进行信用分配的问题。

**核心创新**:将全局 Q 值定义为各智能体 Q 值的**简单求和**。

**关键公式**:

Q_tot(s, a) = sum_{i=1}^{n} Q_i(o_i, a_i)

**IGM 保证**:求和分解天然满足 IGM:
argmax_a sum_i Q_i(o_i, a_i) = (argmax_{a_1} Q_1(o_1, a_1), ..., argmax_{a_n} Q_n(o_n, a_n))

因为对 a_i 取 argmax 时,其他项都是常数。

**训练**:全局 Q_tot 用团队奖励 r 训练:
L = (r + gamma * max_{a'} Q_tot(s', a') - Q_tot(s, a))^2

梯度会自动传到每个 Q_i,实现信用分配。

**优缺点**:
- 优点:保证 IGM,梯度直接传递到各智能体,信用分配清晰
- 缺点:表达能力受限 -- 只能表示**线性**值分解。无法捕捉智能体之间的非线性交互
- 局限:如果最优策略需要非线性的联合动作值,VDN 无法表示

**演化位置**:值分解框架的奠基之作。QMIX 在此基础上引入非线性混合。

### B.3 QMIX (Monotonic Value Function Factorisation)

**来源与动机**:Rashid et al., ICML 2018, Oxford。VDN 的线性分解限制了表达能力。QMIX 引入**单调混合网络**,在保持 IGM 的前提下大幅提升表达能力。

**核心创新**:用一个**单调性约束的混合网络**将各智能体的 Q 值非线性地组合为全局 Q 值。

**关键公式/架构**:

Q_tot(s, a) = f_mix(Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_n(o_n, a_n); s)

其中 f_mix 是一个混合网络,约束为:

dQ_tot / dQ_i >= 0, 对所有 i

这个单调性约束通过**非负权重**实现:混合网络的所有权重由超网络(hypernetwork)从全局状态 s 生成,并经过 softplus 或 abs 确保非负。

架构:
```
Q_1(o_1, a_1) ----\
Q_2(o_2, a_2) ----- [Mixing Network, weights >= 0] ---- Q_tot(s, a)
...             ----/        ^
Q_n(o_n, a_n) ----/         |
                     Hypernetwork(state s)
```

超网络接收全局状态 s,输出混合网络的权重(确保非负)和偏置(无约束)。

**IGM 保证**:
因为 dQ_tot/dQ_i >= 0,Q_tot 关于每个 Q_i 单调递增,所以 argmax 操作可以分散化。

**优缺点**:
- 优点:比 [[05-应用领域/Multi-Agent/QMIX|VDN]] 表达力强得多,可以利用全局状态调节分解方式,在 SMAC 上大幅超越 [[05-应用领域/Multi-Agent/QMIX|VDN]]
- 缺点:单调性约束仍然限制了表达能力 -- 无法表示某些需要非单调分解的值函数(如某些反协调场景)
- Rashid et al. (2020) 的 Weighted [[05-应用领域/Multi-Agent/QMIX|QMIX]] 后续工作分析了这个局限

**演化位置**:值分解方法的里程碑。成为后续大量工作的基线和比较对象。SMAC benchmark 上的标准方法。

### B.4 Weighted [[05-应用领域/Multi-Agent/QMIX|QMIX]] (2020)

**来源与动机**:Rashid et al., 2020, Oxford。[[05-应用领域/Multi-Agent/QMIX|QMIX]] 的单调性约束在某些情况下会导致**投影偏差** -- 即使拿到了最优 Q*,[[05-应用领域/Multi-Agent/QMIX|QMIX]] 的投影也可能恢复不了最优策略。

**核心创新**:在 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 的损失函数中引入**加权**,让更好的联合动作得到更高的拟合优先级。

**关键分析**:

[[05-应用领域/Multi-Agent/QMIX|QMIX]] 可以看作一个投影算子:先计算 Q-learning target,然后投影到 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 可表示的空间中。这个投影在所有联合动作上等权地最小化平方误差,因此可能对"坏"动作的拟合更准确,反而忽略了"好"动作。

**两种加权方案**:
- **CW-[[05-应用领域/Multi-Agent/QMIX|QMIX]] (Centrally-Weighted)**:用一个中心化网络估计的 Q 值作为权重
- **OW-[[05-应用领域/Multi-Agent/QMIX|QMIX]] (Optimistically-Weighted)**:用乐观估计作为权重

两者都证明了可以从任意 Q 值中恢复最优策略。

**优缺点**:
- 优点:理论分析深入,解决了 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 的投影偏差问题,在 predator-prey 和 SMAC 上进一步提升
- 缺点:需要额外的中心化网络,计算开销增加

**演化位置**:[[05-应用领域/Multi-Agent/QMIX|QMIX]] 的理论完善,揭示了值分解方法表达能力与优化目标之间的微妙关系。

### B.5 QPLEX (Dueling Architecture for Value Decomposition)

**来源与动机**:Wang et al., ICLR 2020, 北大。[[05-应用领域/Multi-Agent/QMIX|QMIX]] 的单调性约束虽然保证了 IGM,但表达能力仍然受限。QPLEX 试图在保持 IGM 的前提下,最大化值分解的表达能力。

**核心创新**:引入 **Dueling 架构**,将全局 Q 值分解为 individual Q-values + advantage 的交互项。

**关键公式**:

Q_tot(s, a) = sum_i Q_i(o_i, a_i) + V_adv(s, a)

其中 V_adv 通过一个特殊的分解结构捕捉智能体之间的交互 advantage,同时保持 IGM。

具体地,QPLEX 将 advantage 分解为:
- individual advantages:每个智能体独立的部分
- pairwise advantages:两两交互的部分
- 通过精心设计的约束保持单调性

**理论结果**:QPLEX 在满足 IGM 的值分解方法中,表达能力是**完备的** -- 即任何满足 IGM 的 Q_tot 都可以被 QPLEX 表示。

**优缺点**:
- 优点:理论上最优的表达能力,在 SMAC 上表现优于 [[05-应用领域/Multi-Agent/QMIX|QMIX]]
- 缺点:架构复杂,实现和调参难度高,实际提升在某些场景下不显著

**演化位置**:值分解方法表达能力的理论上界。与 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 共同构成了"简单有效 vs 理论完备"的经典对照。

### B.6 QATT (Attention-based Mixing)

**来源与动机**:Yang et al., 2021。[[05-应用领域/Multi-Agent/QMIX|QMIX]] 的混合网络对所有智能体一视同仁,但不同状态下不同智能体的重要性应该不同。QATT 引入注意力机制来动态调节混合权重。

**核心创新**:用**注意力机制**替代 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 的全连接混合网络,让混合权重动态反映各智能体在当前状态下的重要性。

**关键架构**:

Q_tot = sum_i w_i(s) * Q_i(o_i, a_i) + b(s)

其中权重 w_i 由注意力机制计算:
w_i = softmax(score(Q_i, s))

通过 attention 的 softmax 和正值约束,天然保证 w_i > 0,满足单调性。

**优缺点**:
- 优点:注意力权重可解释,能动态关注重要智能体,在大规模多智能体场景中更具优势
- 缺点:注意力计算随智能体数量增加,在某些 SMAC 场景上提升有限

**演化位置**:将注意力机制引入值分解的开创性工作,启发了后续的 Transformer-based MARL 方法(如 MAT, 2022)。

### 值分解演化小结

```
IQL (1993) -- 无分解,各自学习
  |
VDN (2017) -- 线性求和分解,奠基
  |
QMIX (2018) -- 非线性单调混合,里程碑
  |
  +-- Weighted QMIX (2020) -- 加权投影修正
  |
  +-- QPLEX (2020) -- Dueling 架构,表达力完备
  |
  +-- QATT (2021) -- 注意力混合
  |
  +-- PPS-QMIX (2024) -- 周期性参数共享加速收敛
  |
  +-- AOAD-MAT (2025) -- Transformer + 动作顺序
```

---

## Part C: [[01-原子/策略梯度|策略梯度]]型 MARL

> **演化主线**: COMA -> [[05-应用领域/Multi-Agent/MADDPG|MADDPG]] -> [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] / IPPO -> HAPPO
> **核心思路**:直接优化策略,而非通过值函数间接推导

### C.1 COMA (Counterfactual Multi-Agent Policy Gradients)

**来源与动机**:Foerster et al., AAAI 2018, Oxford。[[01-原子/策略梯度|策略梯度]]方法在 MARL 中的核心困难是**信用分配** -- 如何用团队奖励给出每个智能体的梯度信号?COMA 提出反事实基线来解决这个问题。

**核心创新**:为每个智能体构建**反事实基线(counterfactual baseline)**,衡量"如果这个智能体换一个动作,结果会怎样"。

**关键公式**:

[[01-原子/策略梯度|策略梯度]]:

del_theta_i J = E[del_theta_i log pi_i(a_i|o_i) * A^COMA_i(s, a)]

其中反事实 advantage:

A^COMA_i(s, a) = Q_tot(s, a) - sum_{a_i'} pi_i(a_i'|o_i) * Q_tot(s, (a_i', a_{-i}))

第二项是**反事实基线**:固定其他智能体动作 a_{-i} 不变,对智能体 i 的所有可能动作求期望值。这个基线精确地回答了"在当前状态下,智能体 i 的动作选择相对于平均水平好多少"。

**优缺点**:
- 优点:信用分配精确,理论优雅,是第一个专门为 MARL 设计的[[01-原子/策略梯度|策略梯度]]方法
- 缺点:需要计算所有 a_i' 对应的 Q_tot(s, (a_i', a_{-i})),在动作空间大时计算量大;需要一个中心化的 critic 网络

**演化位置**:[[01-原子/策略梯度|策略梯度]] MARL 的开创性工作。后续 [[05-应用领域/Multi-Agent/MADDPG|MADDPG]] 和 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 都继承了"中心化 critic"的核心思想。

### C.2 [[05-应用领域/Multi-Agent/MADDPG|MADDPG]] (Multi-Agent Deep Deterministic Policy Gradient)

**来源与动机**:Lowe et al., NeurIPS 2017, OpenAI + DeepMind。将 [[02-模块/Actor-Critic/DDPG|DDPG]] 扩展到多智能体,同时解决非平稳性问题。

**核心创新**:每个智能体有一个**中心化 critic** 和一个**分散式 actor**。critic 可以看到全局信息,o_i + (o_j, a_j)_{j!=i},actor 只看本地观测。

**关键架构**:

- Actor: pi_i(o_i) -> a_i (分散式,执行时只依赖本地观测)
- Critic: Q_i(s, a_1, ..., a_n) (集中式,训练时使用全局信息)

critic 的更新:

L = E[(Q_i(s, a) - y)^2], y = r_i + gamma * Q_i'(s', a_1', ..., a_n')

actor 的更新:

del_theta_i J = -del_theta_i Q_i(s, a_1, ..., pi_i(o_i), ..., a_n)

**优缺点**:
- 优点:CTDE 的经典实现,在 MPE(Particle World)环境中表现优异,可以处理竞争和合作设定
- 缺点:基于 [[02-模块/Actor-Critic/DDPG|DDPG]],继承了其对超参数敏感、训练不稳定的缺点;critic 的输入维度随智能体数量线性增长,扩展性差

**演化位置**:CTDE + [[01-原子/策略梯度|策略梯度]]的标志性工作。[[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 本质上是用 [[02-模块/Policy-Based/PPO|PPO]] 替换了 [[05-应用领域/Multi-Agent/MADDPG|MADDPG]] 的 [[02-模块/Actor-Critic/DDPG|DDPG]] 框架。

### C.3 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] (Multi-Agent [[02-模块/Policy-Based/PPO|PPO]])

**来源与动机**:Yu et al., NeurIPS 2022 (Datasets and Benchmarks Track), UC Berkeley + 清华。论文标题直接说明了一切:"The Surprising Effectiveness of [[02-模块/Policy-Based/PPO|PPO]] in Cooperative, Multi-Agent Games"。

**核心创新**:把 [[02-模块/Policy-Based/PPO|PPO]] -- 一个简单、通用的 on-policy 算法 -- 应用到多智能体,配合一些工程上的最佳实践,就能达到甚至超过复杂的 off-policy 方法。

**关键发现**:

1. **[[02-模块/Policy-Based/PPO|PPO]] 在 MARL 中被低估了**:社区普遍认为 on-policy 方法的样本效率不如 off-policy(如 [[05-应用领域/Multi-Agent/QMIX|QMIX]]),但 Yu et al. 通过系统实验表明这是误解

2. **关键工程技巧**使 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 成为强基线:
   - **Value normalization**:对 value 做 running normalization
   - **Advantage normalization**:per-agent advantage 归一化
   - **Large batch size**:大批量训练
   - **ReLU activation**:简单激活函数
   - **Orthogonal initialization**:正交初始化
   - **Clipped value loss**:[[02-模块/Policy-Based/PPO|PPO]] 标准的 clip
   - **Global state to critic**:critic 使用全局状态而非本地观测

3. **在四个 benchmark 上表现强劲**:
   - SMAC (StarCraft):超越 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 等方法
   - MPE (Particle World):竞争力表现
   - Google Research Football:强基线
   - Hanabi:竞争力表现

**关键公式**:

每个智能体 i 维护 actor pi_i 和 critic V_i:

Actor loss ([[02-模块/Policy-Based/PPO|PPO]] clip):
L_actor = -E[min(ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i)]

Critic loss (共享 critic 或独立 critic):
L_critic = E[(V_i(s) - R_i)^2]

ratio_i = pi_i(a_i|o_i) / pi_i_old(a_i|o_i)

**优缺点**:
- 优点:简单、通用、稳定,不需要值分解的结构性约束,在多种环境上表现强,成为事实上的标准基线
- 缺点:样本效率低于 off-policy 方法(on-policy 固有限制),在某些需要精细信用分配的任务上不如 [[05-应用领域/Multi-Agent/QMIX|QMIX]]

**演化位置**:挑战了社区的"复杂度偏见" -- 简单的方法加上好的工程实践可以匹敌复杂方法。成为后续 MARL 研究的标准基线。

### C.4 IPPO (Independent [[02-模块/Policy-Based/PPO|PPO]])

**来源与动机**:de Witt et al., ICML 2021, Oxford。如果 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 的发现是"[[02-模块/Policy-Based/PPO|PPO]] 在多智能体中意外地强",那 IPPO 的发现更极端:**完全独立的 [[02-模块/Policy-Based/PPO|PPO]],不用任何全局信息,也能有竞争力**。

**核心创新**:每个智能体独立运行 [[02-模块/Policy-Based/PPO|PPO]],不共享任何信息。没有中心化 critic,没有全局状态。

**关键发现**:

- 在 MPE 和 SMAC 的多个任务上,IPPO 的表现与 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 接近,有时甚至更好
- 这个结果挑战了 CTDE 的必要性假设 -- 在某些场景下,完全分散式学习已经足够
- 论文强调了**环境设计**和**奖励塑形**对 MARL 性能的影响可能大于算法本身

**优缺点**:
- 优点:最简实现,完美分散,无通信开销,在某些环境中足够好
- 缺点:在需要精细协调的复杂任务上表现不如 CTDE 方法,无法利用训练时的全局信息

**演化位置**:对 CTDE 范式的"挑战者"。提醒研究者:不要默认复杂性是必要的,先用最简单的基线试试。

### C.5 HAPPO (Heterogeneous Agent [[02-模块/Policy-Based/PPO|PPO]])

**来源与动机**:Kuba et al., ICLR 2022。[[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 假设所有智能体是同构的(相同的观测空间、动作空间、网络架构),但现实场景中智能体往往是异构的(不同类型的单位、不同的能力)。

**核心创新**:为异构智能体设计了**序列更新策略**,确保策略更新时不会相互干扰。

**关键思想**:

- 异构智能体无法共享网络参数(因为观测/动作空间不同)
- 简单的独立更新可能导致策略更新的相互干扰
- HAPPO 引入了**序列策略更新**:按顺序更新各智能体的策略,每次更新时考虑之前已更新的策略
- 使用**[[01-原子/重要性采样|重要性采样]]比率**来校正不同智能体之间的策略差异

**关键公式**:

更新智能体 i 时,使用前面已更新智能体的最新策略:

L_i = E[min(ratio_i * A_i, clip(ratio_i) * A_i)]

其中 ratio 考虑了序列更新的顺序效应。

**优缺点**:
- 优点:适用于异构智能体,理论保证收敛,在异构环境(如 SMAC 中混合单位)中优于 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]]
- 缺点:序列更新增加了训练时间,实现复杂度高于 [[05-应用领域/Multi-Agent/MAPPO|MAPPO]]

**演化位置**:[[05-应用领域/Multi-Agent/MAPPO|MAPPO]] 到异构场景的自然扩展。完善了 [[02-模块/Policy-Based/PPO|PPO]]-based MARL 的方法家族。

### [[01-原子/策略梯度|策略梯度]]演化小结

```
COMA (2018) -- 反事实基线,信用分配先驱
  |
MADDPG (2017) -- CTDE + DDPG,中心化 critic
  |
  +-- MAPPO (2022) -- PPO + CTDE,意外的强基线
  |     |
  |     +-- IPPO (2021) -- 独立 PPO,更意外的强基线
  |     |
  |     +-- HAPPO (2022) -- 异构 PPO
  |
  +-- MAT (2022) -- Multi-Agent Transformer
        |
        +-- AOAD-MAT (2025) -- 加入动作顺序
```

---

## Part D: 通信与涌现行为

> **核心问题**:智能体如何通过交换信息来提高协作?能否自发产生通信协议?

### D.1 CommNet (Learning to Communicate)

**来源与动机**:Sukhbaatar et al., NeurIPS 2016, Facebook AI Research。首次系统地研究深度 RL 中智能体如何**学习通信协议**。

**核心创新**:引入一个可微的**通信通道**,让智能体在学习过程中自动发现什么信息值得分享。

**关键架构**:

每个智能体 i 的内部状态 h_i 不仅受自身观测影响,还接收其他智能体的信息:

h_i^(t+1) = f(h_i^t, o_i^t, c_i^t)

其中 c_i 是通信向量:
c_i = sum_{j!=i} h_j^t  (或加权求和)

通信发生在每一层的 hidden state 之间,通过端到端训练,网络自动学习"说什么"和"听谁的"。

**优缺点**:
- 优点:通信协议是学习出来的而非手工设计的,在简单协作任务(如交通管理)上有效
- 缺点:通信是全局广播式的(所有智能体看到所有),可扩展性差;通信内容不可解释

**演化位置**:通信型 MARL 的奠基工作。后续 TarMAC 和 IC3Net 分别在通信目标选择和通信开关上做了改进。

### D.2 IC3Net (Gated Communication)

**来源与动机**:Singh et al., 2018, Facebook AI Research。CommNet 假设通信总是有益的,但实际上在某些场景(如竞争)中,通信可能有害。IC3Net 引入**门控通信**,让智能体自己决定是否通信。

**核心创新**:每个智能体有一个**通信门(gate)**,控制是否向其他智能体发送信息。

**关键架构**:

g_i = sigmoid(W_g * h_i)  // 通信门
c_i = g_i * h_i            // 门控后的通信向量

- g_i 接近 0:不通信
- g_i 接近 1:完全通信
- 门控值通过端到端训练自动学习

**优缺点**:
- 优点:可以选择性地通信,减少不必要的通信开销,在混合(合作+竞争)场景中表现更好
- 缺点:门控函数的训练可能不稳定,在某些环境中门控会退化为全开或全关

**演化位置**:通信型 MARL 中引入"选择性通信"的关键工作。

### D.3 TarMAC (Targeted Multi-Agent Communication)

**来源与动机**:Das et al., ICML 2019, Google Brain。CommNet 的全局广播和 IC3Net 的二元门控都太粗糙了。TarMAC 引入**注意力机制**实现精确的目标通信。

**核心创新**:每个智能体不仅决定"说什么",还决定"对谁说"。使用注意力机制让发送者定向选择接收者。

**关键架构**:

智能体 i 发送消息 m_i,并生成一个 query q_i。
智能体 j 生成一个 key k_j。
注意力权重:
w_{ij} = softmax_j(q_i * k_j)

智能体 i 接收的信息:
c_i = sum_j w_{ij} * m_j

这样,智能体 i 可以精准地从最相关的智能体那里获取信息。

**优缺点**:
- 优点:精确的目标通信,减少噪声,注意力权重可解释,在大规模场景中更具优势
- 缺点:注意力计算 O(n^2),在智能体数量很多时仍有开销

**演化位置**:通信型 MARL 中引入注意力机制的标志性工作。与后续的 Transformer-based MARL(MAT)有直接的思想联系。

### D.4 涌现通信 (Emergent Communication)

**来源与动机**:如果完全不给智能体预定义通信协议,它们能否自发"发明"一种语言?这是 AI 和语言学交叉的前沿问题。

**核心概念**:

涌现通信研究的是:多个智能体在需要协作完成任务时,能否通过 RL 自发产生一套有意义的符号系统?

经典实验:
- ** referential game**:发送者看到目标对象,需要通过通信通道让接收者从多个候选中选出目标
- **Lewis Signaling Game**:发送者观察状态,发送离散符号,接收者根据符号选择动作

**关键发现**:

1. **符号确实会涌现**:智能体会自发产生离散的符号系统来传递信息
2. **但涌现的"语言"往往不是组合性的**:和人类语言不同,涌现的符号系统通常是"整体式"的 -- 每个符号对应一个完整的意义,而非由词根和语法组合
3. **组合性需要特殊条件**:要涌现出类似人类语言的组合性结构,通常需要:
   - 足够复杂的任务
   - 通信通道的信息瓶颈
   - 任务组合性压力

4. **与语言游戏的联系**:Van Eecke & Beuls (2020) 提出将语言游戏范式与 MARL 结合,利用 MARL 的方法论推进涌现通信研究

**优缺点**:
- 优点:探索了 AI 通信的极限,对理解人类语言起源有启发意义
- 缺点:涌现的协议不可解释、不可泛化,距离实用还有很大距离

**演化位置**:MARL 与认知科学、语言学的交叉前沿。2024-2026 年 LLM 智能体的出现为此方向注入了新活力。

---

## Part E: 高级主题与里程碑系统

### E.1 Population-Based Training (PBT) 与联赛训练

**来源与动机**:Jaderberg et al., 2019, DeepMind。传统超参数调优和模型选择是分离的。PBT 将两者统一:在训练过程中同时优化超参数和模型参数。

**核心思想**:

1. 维护一个**种群(population)** 的模型,每个模型有不同的超参数
2. 训练过程中,**表现差**的模型被**替换**为表现好的模型的副本,但带有随机扰动的超参数
3. 类似于进化算法的 **exploit**(利用好的) + **explore**(探索新的)

**联赛训练 (League Training)**:

AlphaStar (Vinyals et al., 2019) 将 PBT 扩展为联赛训练:

```
Main Agents (主力)
  |--- 与所有对手训练
  |
Main Exploiters (主力剥削者)
  |--- 专门训练如何击败 Main Agents
  |
League Exploiters (联赛剥削者)
  |--- 专门训练如何击败整个联赛的弱点
```

- **Main Agents**:追求最终胜率
- **Main Exploiters**:防止 Main Agents 过拟合到特定策略
- **League Exploiters**:发现整个联赛的系统性弱点

这种结构避免了**策略循环**(A 克 B, B 克 C, C 克 A)导致的训练不稳定。

**优缺点**:
- 优点:自动超参数调优,避免过拟合,产生多样化的策略
- 缺点:计算资源需求巨大(AlphaStar 用了数千个 TPU 天)

**演化位置**:Self-play -> PBT -> League Training,是大规模对抗训练的范式进化。

### E.2 Self-Play (自博弈)

**来源与动机**:Self-play 的思想可以追溯到博弈论中的 fictitious play,但在深度 RL 中的成功主要来自 DeepMind 的 AlphaGo/AlphaZero/AlphaStar 系列。

**核心思想**:智能体通过与自身的副本对弈来训练。每代智能体都是上一代的改进版本,训练难度自动递增。

**演化历程**:

1. **AlphaGo (2016)**:先用人类数据预训练,再通过 self-play 强化学习
2. **AlphaGo Zero / AlphaZero (2017)**:完全从零开始,纯 self-play + MCTS
3. **AlphaStar (2019)**:将 self-play 扩展为 league training + 人口统计

**Self-play 的关键问题**:

- **策略循环**:A > B > C > A,导致训练震荡
- **灾难性遗忘**:学会对付新对手后,忘记如何对付旧对手
- **策略多样性**:self-play 可能收敛到单一策略,无法应对多样化的对手

League training 和 population-based training 都是为解决这些问题而设计的。

### E.3 OpenAI Five (Dota 2)

**来源与动机**:OpenAI, 2019。目标是在 Dota 2 -- 世界上最复杂的电子竞技游戏之一 -- 中击败职业选手。

**核心架构**:

- 5 个独立的 LSTM 策略网络(每个英雄一个)
- 使用 [[02-模块/Policy-Based/PPO|PPO]] 训练
- **极其夸张的规模**:
  - 128,000 CPU 核心
  - 256 GPU
  - 每天自我对弈 180 年游戏经验(加速模拟)
  - 持续训练 10 个月

**关键设计**:

- **共享团队奖励**:5 个智能体共享同一个奖励信号(胜负)
- **Courier(信使)系统**:独立的信使智能体负责物品运输
- **Action space**:每个英雄约 1.7 * 10^4 种可能动作
- **Observation space**:约 20,000 维度的状态向量
- **LSTM**:1024 维度的 LSTM 处理序列信息

**结果**:2019 年 4 月在 BO3 中 2:1 击败世界冠军 OG。

**教训与局限**:
- 展示了**规模暴力(scale)** 在 MARL 中的威力
- 但也暴露了局限:对游戏版本更新脆弱,策略多样性有限
- 5 个智能体实际是**同质**的(都是 [[02-模块/Policy-Based/PPO|PPO]] + LSTM),没有利用到异构性

**演化位置**:证明了 [[02-模块/Policy-Based/PPO|PPO]] + 大规模计算可以解决极高复杂度的多智能体问题。

### E.4 AlphaStar (StarCraft II)

**来源与动机**:Vinyals et al., Nature 2019, DeepMind。在星际争霸 II -- 即时战略游戏的巅峰 -- 中达到 Grandmaster 水平。

**核心架构(远比 OpenAI Five 复杂)**:

1. **监督学习预训练**:从人类 replay 中学习基础策略
2. **强化学习微调**:通过 self-play 超越人类水平
3. **League Training**:联赛训练避免策略循环
4. **Population of agents**:维护一个多样化的策略池

**关键技术**:

- **模仿学习(Imitation Learning)**:从人类 replay 学习,提供初始策略
- **Multi-agent RL**:联赛中的每个 agent 都是一个独立的学习者
- **Distillation**:将复杂策略蒸馏为更小的网络
- **Fingerprints**:用对手的历史特征来识别其策略类型

**结果**:在 StarCraft II 天梯中达到 Grandmaster(排名前 0.15%)。

**与 OpenAI Five 的对比**:

| | OpenAI Five (Dota 2) | AlphaStar (StarCraft II) |
|---|---|---|
| 训练方法 | [[02-模块/Policy-Based/PPO|PPO]] from scratch | SL pretrain + RL + League |
| 复杂度 | 5 个同质 agent | 多种族异构 agent |
| 对手多样性 | 纯 self-play | League training + population |
| 人类数据 | 不用 | 大量使用 replay |
| 架构 | LSTM | Transformer + LSTM |

**演化位置**:MARL 里程碑。展示了如何将 SL + RL + population-based training 有机融合,是后续大规模 MARL 系统的蓝图。

### E.5 CICERO (Diplomacy)

**来源与动机**:Bakhtin et al., Science 2022, Meta AI。Diplomacy(外交棋)是一个需要语言沟通、结盟、欺骗和战略规划的策略棋盘游戏。

**核心创新**:将**语言模型**、**规划模块**和**强化学习**整合成一个统一的系统。

**关键架构**:

1. **语言模型(dialogue model)**:基于 GPT 微调的语言模型,生成游戏中的对话(谈判、结盟、欺骗)
2. **规划模块(planning)**:
   - **PiKL(Pi Best Response with KL)**:基于人类倾向的先验策略
   - 通过搜索找到最优动作组合
3. **价值函数**:评估当前局势,训练自 self-play
4. **意图识别**:从对话中推断其他玩家的意图

**CICERO 的运作流程**:

```
当前局势 + 对话历史
     |
     v
[语言模型] -> 生成对话(谈判/欺骗)
     |
     v
[意图推断] -> 预测其他玩家可能的动作
     |
     v
[规划模块] -> 搜索最优动作
     |
     v
[执行动作 + 发送消息]
```

**结果**:在 webDiplomacy 上达到人类水平(排名前 ~10%)。

**局限**(Wongkamjan et al., ACL 2024):
- 后续分析表明 CICERO 在欺骗和说服方面的能力有限
- 更多依赖策略能力而非沟通能力
- 标注分析显示 AI-人类通信仍然受限

**演化位置**:第一个成功将语言与策略结合的大规模 MARL 系统。预示了 2024-2026 LLM-based multi-agent 的方向。

### E.6 主要 MARL Benchmark

**SMAC (StarCraft Multi-Agent Challenge)**
- Rashid et al., 2019
- 星际争霸微操场景,控制一支小部队歼灭敌军
- 3-27 个智能体,纯合作
- 多种难度场景(3s5z, 2c_vs_64zg, corridor 等)
- 是值分解方法([[05-应用领域/Multi-Agent/QMIX|QMIX]] 等)的标准测试场

**MPE (Multi-Agent Particle Environments)**
- Lowe et al., 2017
- 2D 粒子世界,包含合作导航、捕猎、通信等任务
- 简单但经典,适合快速验证
- 包含合作、竞争和混合场景

**PettingZoo**
- Terry et al., 2021
- Gymnasium 风格的多智能体环境库
- 包含经典博弈(囚徒困境、拍卖)、Atari 多智能体、棋盘游戏等
- 标准化 API(AEC: Alternate Environment Cycle)
- 社区维护,持续扩展

**Melting Pot**
- Leibo et al., 2021, DeepMind
- 专注于**社会困境**和**混合动机**场景
- 强调泛化性:训练后的智能体需要在未见过的新环境中与新对手交互
- 包含公共物品博弈、资源采集、领地争夺等场景
- 2023 年发布 Melting Pot 2.0,场景更加丰富

**Hanabi**
- Bard et al., 2019, DeepMind
- 合作卡牌游戏,信息受限(看不到自己的手牌)
- 强调隐式通信和理论心智(theory of mind)
- 2-5 人,难度随人数指数增长
- SAD(Simplified Action Decoder) 等方法是此 benchmark 上的 SOTA

---

## Part F: 2024-2026 前沿

### F.1 LLM-based Multi-Agent Systems

2024-2026 年,大语言模型(LLM)的崛起为多智能体系统注入了全新的维度。

**LLM 作为智能体(LLM-as-Agent)**:

传统 MARL 中的智能体是小型神经网络(MLP/LSTM),通过 RL 从零学习策略。LLM-based agent 则利用预训练语言模型作为智能体的"大脑":

- 具备常识推理能力
- 可以进行自然语言通信
- 能够从指令中学习,不需要大量试错
- 可以调用工具(tool use)

**代表性方向**:

1. **LLM 辩论与协作**:
   - Du et al. (2023) "Improving Factuality and Reasoning via Multiagent Debate":多个 LLM agent 通过辩论提高推理质量
   - Li et al. (2023) "CAMEL":角色扮演框架下的 LLM 协作

2. **LLM 社会模拟**:
   - Park et al. (2023) "Generative Agents":25 个 LLM agent 在虚拟小镇中生活,展现出涌现的社会行为
   - 后续工作扩展到更大规模的社会模拟

3. **LLM + MARL 训练**:
   - 用 LLM 作为 MARL 智能体的通信接口
   - LLM 提供高层规划和推理,RL 提供底层执行策略
   - 结合 CICERO 的思路:语言 + 规划 + RL

4. **LLM agent 在博弈论环境中的表现**:
   - 研究 LLM 在囚徒困境、拍卖等场景中的策略行为
   - LLM 展现出"类人"的合作倾向和公平偏好
   - 但也暴露了对欺骗和策略博弈的有限能力

### F.2 后 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 时代的值分解

2024-2026 年的值分解研究方向已经从"如何更好地分解 Q 值"转向更根本的问题:

1. **Transformer-based MARL**:
   - **MAT (Multi-Agent Transformer, 2022)**:将 Transformer 引入 MARL 的 actor-critic 架构
   - **AOAD-MAT (Takayama & Fujita, PRIMA 2025)**:在 MAT 基础上加入动作决策顺序建模,在 SMAC 和 MuJoCo 上超越 MAT

2. **通信效率**:
   - **IA-KRC (Cheng et al., 2026)**:K-step 可达通信,通过干扰感知选择通信对象,在动态拓扑中实现高效协作
   - 强调在带宽受限环境下的智能通信选择

3. **参数共享与个性化**:
   - **PPS-[[05-应用领域/Multi-Agent/QMIX|QMIX]] (Zhang et al., 2024)**:受联邦学习启发,周期性参数共享加速 [[05-应用领域/Multi-Agent/QMIX|QMIX]] 训练,在 SMAC 上提升 10-30%

### F.3 安全与对齐

**SMARL (Shielded MARL)**:
- Chatterji & Acar, ECAI 2025
- 将概率逻辑盾(Probabilistic Logic Shields)引入 MARL
- 在去中心化多智能体环境中强制执行安全约束
- 提出 PLTD(Probabilistic Logic TD)更新和带安全保证的[[01-原子/策略梯度|策略梯度]]方法

### F.4 团队形成与动态种群

**双边团队形成 (Bilateral Team Formation)**:
- Moslemi & Lee, RLC 2025 CoCoMARL Workshop
- 研究动态种群中的双边(而非单方面)团队组建
- 关注算法属性如何影响策略性能和泛化能力

### F.5 关键趋势总结

1. **LLM + MARL 融合**:LLM 作为智能体的"大脑"或通信接口,RL 提供底层策略优化。这是 2024-2026 最重要的趋势。

2. **从 benchmark 竞赛到真实应用**:SMAC/MPE 上的性能竞赛逐渐让位于更现实的应用场景(交通、能源、金融)。

3. **Transformer 架构渗透**:MAT、AOAD-MAT 等工作将 Transformer 的序列建模能力引入 MARL。

4. **安全与对齐**:随着 MARL 走向应用,安全性和规范性约束成为新关注点(SMARL)。

5. **通信的精细化**:从"全广播"到"精确目标通信"到"带宽感知通信",通信机制越来越精细。

---

## 附录:核心论文速查表

| 方法 | 年份 | 作者 | 发表 | 类型 | 核心贡献 |
|---|---|---|---|---|---|
| [[02-模块/Value-Based/IQL|IQL]] | 1993 | Tan | -- | 基线 | 独立学习的朴素基线 |
| [[05-应用领域/Multi-Agent/QMIX|VDN]] | 2017 | Sunehag et al. | -- | 值分解 | 线性求和分解 |
| [[05-应用领域/Multi-Agent/QMIX|QMIX]] | 2018 | Rashid et al. | ICML | 值分解 | 单调混合网络 |
| [[05-应用领域/Multi-Agent/MADDPG|MADDPG]] | 2017 | Lowe et al. | NeurIPS | [[01-原子/策略梯度|策略梯度]] | CTDE + [[02-模块/Actor-Critic/DDPG|DDPG]] |
| COMA | 2018 | Foerster et al. | AAAI | [[01-原子/策略梯度|策略梯度]] | 反事实基线 |
| CommNet | 2016 | Sukhbaatar et al. | NeurIPS | 通信 | 端到端通信学习 |
| IC3Net | 2018 | Singh et al. | -- | 通信 | 门控通信 |
| TarMAC | 2019 | Das et al. | ICML | 通信 | 注意力目标通信 |
| QPLEX | 2020 | Wang et al. | ICLR | 值分解 | Dueling 表达力完备 |
| WQMIX | 2020 | Rashid et al. | NeurIPS | 值分解 | 加权投影修正 |
| QATT | 2021 | Yang et al. | -- | 值分解 | 注意力混合 |
| IPPO | 2021 | de Witt et al. | ICML | [[01-原子/策略梯度|策略梯度]] | 独立 [[02-模块/Policy-Based/PPO|PPO]] |
| [[05-应用领域/Multi-Agent/MAPPO|MAPPO]] | 2022 | Yu et al. | NeurIPS | [[01-原子/策略梯度|策略梯度]] | [[02-模块/Policy-Based/PPO|PPO]] 的意外强力 |
| HAPPO | 2022 | Kuba et al. | ICLR | [[01-原子/策略梯度|策略梯度]] | 异构 [[02-模块/Policy-Based/PPO|PPO]] |
| CICERO | 2022 | Bakhtin et al. | Science | 系统 | 语言+规划+RL |
| MAT | 2022 | Wen et al. | ICLR | 架构 | Multi-Agent Transformer |
| AlphaStar | 2019 | Vinyals et al. | Nature | 系统 | League training |
| OpenAI Five | 2019 | OpenAI | -- | 系统 | 大规模 [[02-模块/Policy-Based/PPO|PPO]] |
| PBT | 2019 | Jaderberg et al. | -- | 训练 | 种群训练 |
| AOAD-MAT | 2025 | Takayama, Fujita | PRIMA | 架构 | 动作顺序 MAT |
| PPS-[[05-应用领域/Multi-Agent/QMIX|QMIX]] | 2024 | Zhang et al. | -- | 值分解 | 周期参数共享 |
| SMARL | 2025 | Chatterji, Acar | ECAI | 安全 | 概率逻辑盾 |
| IA-KRC | 2026 | Cheng et al. | -- | 通信 | K-step 可达通信 |

---

## 附录:方法演化全景图

```
                    MARL 方法演化
                         |
           +-------------+-------------+
           |                           |
      值分解族                       策略梯度族
           |                           |
      IQL (1993)               COMA (2018)
           |                    /         \
      VDN (2017)        MADDPG (2017)    ...
           |                 |
      QMIX (2018)       MAPPO (2022)
       /  |  \              |  \
  WQMIX QPLEX QATT     IPPO  HAPPO
 (2020)(2020)(2021)   (2021) (2022)
                           |
                        MAT (2022)
                           |
                      AOAD-MAT (2025)

           |                           |
      通信族                         系统族
           |                           |
   CommNet (2016)            AlphaStar (2019)
           |                  OpenAI Five (2019)
   IC3Net (2018)              CICERO (2022)
           |
   TarMAC (2019)
           |
   IA-KRC (2026)
```

---

*本文档综合了 arXiv、Google Scholar 和已发表论文的调研结果。所有方法的核心公式和架构描述基于原始论文。2024-2026 前沿部分基于最新 arXiv 搜索结果(截至 2026-06-21)。*
