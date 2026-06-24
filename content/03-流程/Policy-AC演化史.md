# [[01-原子/策略梯度|策略梯度]]与 Actor-Critic 强化学习：完整演化史

> 从 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] (1992) 到 [[02-模块/Policy-Based/DAPO|DAPO]] (2025)，覆盖策略优化方法的核心脉络。
> 每个方法按「来源与动机 → 核心创新 → 关键公式 → 优缺点 → 演化位置」展开。

---

## 演化全景图

```
                        策略梯度方法演化树
                        
    离散/通用控制                              连续控制                    LLM 对齐
    ─────────────                          ──────────                ──────────
    
    REINFORCE (1992)                       DDPG (2016)              PPO-RLHF (2022)
        │                                      │                        │
        ▼                                      ▼                        ▼
    Policy Gradient Thm (1999)             TD3 (2018)              DPO (2023)
        │                                      │                        │
        ▼                                      ▼                        ▼
    Actor-Critic (1983/2000)               SAC (2018)              GRPO (2024)
        │                                                            │
        ▼                                                            ▼
    Natural PG (2002)                                           DAPO (2025)
        │                                                            │
        ▼                                                            ▼
    TRPO (2015)                                                 DCPO (2025)
        │
        ▼
    A3C/A2C (2016)
        │
        ▼
    PPO (2017)
```

---

## 一、[[02-模块/Policy-Based/REINFORCE|REINFORCE]]：[[01-原子/策略梯度|策略梯度]]的开山之作

### 来源与动机

- **论文**：Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.
- **解决的问题**：在 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] 之前，强化学习主要靠 TD-learning 和 Q-learning 等值函数方法。这些方法在离散动作空间表现不错，但无法直接优化参数化策略。Williams 提出：**能不能直接对策略本身求梯度？**

### 核心创新

> 用蒙特卡洛回报作为[[01-原子/策略梯度|策略梯度]]的无偏估计——"采样一条轨迹，按回报加权，增大好动作的概率"。

### 关键公式

[[01-原子/策略梯度|策略梯度]]估计：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的折扣回报。

### 优缺点

**优点**：
- 概念极简——直接对策略求导，不需要值函数
- 可处理连续动作空间和随机策略
- 理论上有无偏性保证

**缺点**：
- **方差极大**：$G_t$ 的方差随轨迹长度指数增长
- 样本效率低——需要大量采样才能得到可靠梯度
- 每步只使用一条轨迹的信息

### 演化位置

[[02-模块/Policy-Based/REINFORCE|REINFORCE]] 是所有[[01-原子/策略梯度|策略梯度]]方法的「始祖」。后续的 Baseline、Actor-Critic、[[02-模块/Policy-Based/TRPO|TRPO]]、[[02-模块/Policy-Based/PPO|PPO]] 全部建立在它的核心思想之上——**沿回报方向增大动作概率**。

---

## 二、Baseline 方法：降低方差的第一个 trick

### 来源与动机

- **核心思想来源**：Weaver, L. C. & Tao, N. (2001). "The optimal reward baseline for gradient-based reinforcement learning." *UAI*.
- **理论完善**：Sutton, McAllester, Singh, Mansour (1999/2000) 在 Policy Gradient Theorem 中形式化。
- **解决的问题**：[[02-模块/Policy-Based/REINFORCE|REINFORCE]] 的梯度估计方差太大，训练极不稳定。

### 核心创新

> 在回报中减去一个与动作无关的基线值 $b(s)$——"不是回报的绝对值重要，而是比平均水平好多少才重要"。

### 关键公式

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t)) \right]$$

### 为什么 Baseline 不引入偏差？

**关键证明**：对任意 $b(s_t)$（只依赖于状态，不依赖于动作），

$$\mathbb{E}_{a_t \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t)] = b(s_t) \cdot \nabla_\theta \sum_{a} \pi_\theta(a|s_t) = b(s_t) \cdot \nabla_\theta 1 = 0$$

因为策略对所有动作的概率求和恒等于 1，其梯度为零。所以减去 $b(s_t)$ 不改变期望值，但可以显著降低方差。

**最优 baseline**（最小方差）：

$$b^*(s) = \frac{\mathbb{E}[G_t^2 \cdot \nabla_\theta \log \pi_\theta]}{\mathbb{E}[G_t \cdot \nabla_\theta \log \pi_\theta]}$$

实践中通常用状态值函数 $V(s)$ 作为 baseline。

### 优缺点

**优点**：
- 不改变梯度期望（无偏），但显著降低方差
- 实现简单——只需要额外估计 $V(s)$
- 是后续所有 Actor-Critic 方法的基础

**缺点**：
- $V(s)$ 本身的估计也有误差
- 单独使用 baseline 仍不足以解决样本效率问题

### 演化位置

Baseline 是 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] → Actor-Critic 的桥梁。当你用 $V(s)$ 作为 baseline 时，你就已经在做 Actor-Critic 了。

---

## 三、Policy Gradient Theorem：理论基石

### 来源与动机

- **论文**：Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (1999/2000). "Policy gradient methods for reinforcement learning with function approximation." *NIPS 1999*.
- **解决的问题**：[[02-模块/Policy-Based/REINFORCE|REINFORCE]] 的梯度估计虽然是正确的，但缺乏严格的理论框架。[[01-原子/策略梯度|策略梯度]]方法在函数近似下是否收敛？

### 核心创新

> 证明了策略性能梯度可以用动作值函数 $Q^\pi$ 的闭式表达——不需要对状态分布求导。

### 关键公式

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a|s) \cdot Q^\pi(s, a)$$

其中 $d^\pi(s)$ 是在策略 $\pi$ 下的状态访问分布（stationary distribution）。

**关键洞察**：梯度只涉及 $\nabla_\theta \pi_\theta$，不涉及 $\nabla_\theta d^\pi(s)$。状态分布对策略参数的依赖"消失"了，这大大简化了问题。

### 优缺点

**优点**：
- 给出了[[01-原子/策略梯度|策略梯度]]的精确解析表达式
- 证明了[[01-原子/策略梯度|策略梯度]]方法的收敛性（在合适的步长条件下）
- 为 Actor-Critic 提供了理论基础（用 $Q$ 的近似替代真实 $Q^\pi$）

**缺点**：
- 理论结果假设了充分探索，实践中难以保证
- 收敛速率分析不够精细

### 演化位置

这是[[01-原子/策略梯度|策略梯度]]方法的「基本定理」。所有后续方法——Actor-Critic、Natural PG、[[02-模块/Policy-Based/TRPO|TRPO]]、[[02-模块/Policy-Based/PPO|PPO]]——都是这个定理的不同实例化。

---

## 四、Actor-Critic：双网络架构的诞生

### 来源与动机

- **原始思想**：Barto, A. G., Sutton, R. S., Anderson, C. W. (1983). "Neuronlike adaptive elements that can solve difficult learning control problems." *IEEE Transactions on Systems, Man, and Cybernetics*.
- **现代理论**：Konda, V. R. & Tsitsiklis, J. N. (2000). "Actor-critic algorithms." *NIPS 1999*.
- **解决的问题**：[[02-模块/Policy-Based/REINFORCE|REINFORCE]] 需要等整条轨迹结束才能计算 $G_t$（蒙特卡洛方法），导致高方差和低效率。能不能用 TD-learning 的思想，每步都更新？

### 核心创新

> 用两个网络——Actor（策略）负责选动作，Critic（值函数）负责评估好坏——Critic 用 TD 更新提供即时反馈，Actor 用 Critic 的评估来更新策略。

### 关键公式

**Actor 更新**（[[01-原子/策略梯度|策略梯度]]）：
$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a|s) \cdot \delta$$

**Critic 更新**（[[01-原子/TD误差|TD 误差]]）：
$$\delta = r + \gamma V_\phi(s') - V_\phi(s)$$
$$\phi \leftarrow \phi + \alpha_\phi \delta \nabla_\phi V_\phi(s)$$

其中 $\delta$ 就是 [[01-原子/TD误差|TD 误差]]，替代了 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] 中的 $G_t$。

### 优缺点

**优点**：
- 每步都能更新（不需要等轨迹结束），方差比 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] 低很多
- 可以处理无限时间步的问题
- Critic 提供了更稳定的学习信号

**缺点**：
- Critic 的估计有偏差（bootstrap），偏差会传播给 Actor
- 两个网络联合训练可能不稳定
- Konda & Tsitsiklis (2000) 证明了线性函数近似下的收敛性，但深度神经网络下收敛性无保证

### 演化位置

Actor-Critic 是[[01-原子/策略梯度|策略梯度]]方法的「工程化转折点」。[[02-模块/Actor-Critic/A3C|A3C]]、[[02-模块/Policy-Based/PPO|PPO]]、[[02-模块/Actor-Critic/DDPG|DDPG]]、[[02-模块/Actor-Critic/SAC|SAC]] 全部是 Actor-Critic 的变体。[[02-模块/Policy-Based/REINFORCE|REINFORCE]] 是纯[[01-原子/策略梯度|策略梯度]]，Actor-Critic 是[[01-原子/策略梯度|策略梯度]] + 值函数近似的混合体。

---

## 五、Natural Policy Gradient：利用信息几何

### 来源与动机

- **论文**：Kakade, S. M. (2002). "A natural policy gradient." *NIPS 2001*.
- **解决的问题**：标准[[01-原子/策略梯度|策略梯度]]用欧几里得梯度 $\nabla_\theta J$ 更新，但参数空间的欧几里得距离不反映策略分布的真实差异。同一个策略用不同参数化表示，欧几里得梯度方向完全不同。

### 核心创新

> 用 Fisher 信息矩阵的逆来"校正"梯度方向——在策略分布的空间里走最短路径，而不是在参数空间里。

### 关键公式

**标准梯度**：
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

**自然梯度**：
$$\theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta)$$

其中 Fisher 信息矩阵：
$$F = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot (\nabla_\theta \log \pi_\theta(a|s))^\top]$$

**直觉**：$F$ 度量了参数变化引起的策略分布变化。$F^{-1}$ 把"参数空间中最陡的方向"转换为"策略空间中最有效的方向"。

### 优缺点

**优点**：
- 参数化无关——同一策略不管用什么网络结构，自然梯度方向一致
- 学习效率更高——避免在"参数变了很多但策略没怎么变"的方向上浪费步长
- 为 [[02-模块/Policy-Based/TRPO|TRPO]] 提供了直接的理论基础

**缺点**：
- $F$ 的计算和求逆代价极高（$n \times n$ 矩阵，$n$ 为参数量）
- 实践中需要用共轭梯度法近似求解 $F^{-1} g$
- 步长选择仍需要手工调节

### 演化位置

Natural PG 是 [[02-模块/Policy-Based/REINFORCE|REINFORCE]] → [[02-模块/Policy-Based/TRPO|TRPO]] 的关键中间环节。[[02-模块/Policy-Based/TRPO|TRPO]] 的核心思想——在策略空间（而非参数空间）约束更新步长——直接来源于 Natural PG 的洞察。

---

## 六、[[02-模块/Policy-Based/TRPO|TRPO]]：单调改进的理论保证

### 来源与动机

- **论文**：Schulman, J., Levine, S., Moritz, P., Jordan, M. I., Abbeel, P. (2015). "Trust region policy optimization." *ICML 2015*. arXiv: 1502.05477.
- **解决的问题**：[[01-原子/策略梯度|策略梯度]]方法更新步长太大时性能可能崩溃（一个坏更新就可能毁掉之前所有的学习）。Natural PG 虽然用 Fisher 信息矩阵校正方向，但没有约束步长大小。

### 核心创新

> 用 KL 散度约束每次策略更新——保证新策略和旧策略"足够接近"，从而数学上证明性能单调不减。

### 关键公式

**优化目标**：
$$\max_\theta \quad \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}} \left[ \sum_a \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \hat{A}^{\pi_{\theta_\text{old}}}(s,a) \right]$$

**约束条件**：
$$\mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}} [D_\text{KL}(\pi_{\theta_\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s))] \leq \delta$$

其中 $\delta$ 是最大允许的 [[01-原子/KL散度|KL 散度]]。

**单调改进保证（定理）**：如果 $\bar{A}^{\pi_{\theta_\text{old}}}(s) \geq 0$ 对所有 $s$ 成立（即[[01-原子/优势函数|优势函数]]非负），且 $D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$，则 $J(\pi_\theta) \geq J(\pi_{\theta_\text{old}})$。

### 为什么 KL 约束重要？

**核心洞察**：策略性能的下界可以用 KL 散度来表达：

$$J(\pi_\theta) \geq J(\pi_{\theta_\text{old}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} D_\text{KL}^{\max}(\pi_{\theta_\text{old}} \| \pi_\theta)$$

其中 $\epsilon$ 是[[01-原子/优势函数|优势函数]]的最大绝对值。KL 散度越小，性能下界越紧。

### 优缺点

**优点**：
- **理论保证**：每次更新保证性能单调不减（在理论精确实现下）
- 对超参数不敏感——$\delta$ 是唯一关键超参数
- 在各种任务上表现稳健

**缺点**：
- **实现复杂**：需要计算 Fisher 向量积、共轭梯度法求解、线搜索
- 每一步需要大量计算（二阶信息）
- 只适用于 on-policy 训练，样本效率一般
- 与值函数近似联合使用时，理论保证不完全成立

### 演化位置

[[02-模块/Policy-Based/TRPO|TRPO]] 是策略优化方法的「理论巅峰」。它的 KL 约束思想直接影响了 [[02-模块/Policy-Based/PPO|PPO]]（简化版 [[02-模块/Policy-Based/TRPO|TRPO]]）和后续的 LLM RL 方法。但工程上的复杂性使得实践中 [[02-模块/Policy-Based/PPO|PPO]] 更受欢迎。

---

## 七、[[02-模块/Policy-Based/PPO|PPO]]：简单即正义

### 来源与动机

- **论文**：Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). "Proximal policy optimization algorithms." arXiv: 1707.06347.
- **解决的问题**：[[02-模块/Policy-Based/TRPO|TRPO]] 效果好但实现太复杂（共轭梯度、线搜索、Fisher 向量积）。能不能用更简单的方法达到类似效果？

### 核心创新

> 用 clip 替代 KL 约束——把概率比限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，一行代码实现信赖域。

### 关键公式

**[[02-模块/Policy-Based/PPO|PPO]]-Clip 目标函数**：

$$\mathcal{J}_\text{PPO}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中概率比 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$。

**Clip 的直觉**：
- 当 $\hat{A}_t > 0$（好动作）：增大 $\pi_\theta(a_t|s_t)$，但不超过 $(1+\epsilon) \cdot \pi_{\theta_\text{old}}$
- 当 $\hat{A}_t < 0$（坏动作）：减小 $\pi_\theta(a_t|s_t)$，但不低于 $(1-\epsilon) \cdot \pi_{\theta_\text{old}}$
- 超出范围后梯度为零——"不鼓励剧烈变化"

### PPO 为什么能替代 TRPO 的 KL 约束？

| 维度 | TRPO | PPO-Clip |
|------|------|----------|
| 约束方式 | KL 散度硬约束 $\leq \delta$ | Clip 软约束 $[1-\epsilon, 1+\epsilon]$ |
| 实现复杂度 | 共轭梯度 + 线搜索 | 一行 `torch.clamp` |
| 理论保证 | 严格单调改进 | 无严格保证（但实践中足够好） |
| 步长控制 | 自适应（线搜索） | 固定 $\epsilon$（通常 0.2） |
| 多次更新 | 每次迭代只用一次数据 | 可以多个 epoch 复用同一批数据 |

**PPO 的核心优势**：可以对同一批数据做多次 SGD 更新（multi-epoch），因为 clip 限制了每次更新的幅度。这大幅提高了样本效率。

### 优缺点

**优点**：
- 实现极简——核心逻辑不到 10 行代码
- 训练稳定，对超参数不太敏感
- 样本效率比 TRPO 好（多 epoch 更新）
- 通用性强——连续/离散动作空间都适用
- 是 LLM RLHF 的标准方法（OpenAI InstructGPT、ChatGPT）

**缺点**：
- 没有 TRPO 的严格单调改进保证
- $\epsilon$ 是固定的，不能自适应
- 在某些任务上不如 SAC 等 off-policy 方法样本效率高
- 对于 LLM RL 场景，存在熵崩溃（entropy collapse）问题（见 DAPO）

### 演化位置

PPO 是当前「最广泛使用的策略优化方法」。从机器人控制到 LLM 对齐，PPO 无处不在。它是 TRPO 的实用简化版，也是 GRPO、DAPO 等 LLM RL 方法的直接前身。

---

## 八、A3C/A2C：并行化加速

### 来源与动机

- **论文**：Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., Kavukcuoglu, K. (2016). "Asynchronous methods for deep reinforcement learning." *ICML 2016*. arXiv: 1602.01783.
- **解决的问题**：DQN 需要经验回放（experience replay）来打破数据相关性，但经验回放需要大容量内存且不支持 on-policy 方法。能不能用并行化替代经验回放？

### 核心创新

> 多个 worker 并行在不同环境实例中收集数据，异步更新共享参数——并行性天然打破数据相关性，无需[[01-原子/经验回放|经验回放]]。

### 关键公式

**[[02-模块/Actor-Critic/A3C|A3C]] 的 Actor-Critic 更新**（每个 worker $i$）：

$$d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \cdot (R - V_{\theta'_v}(s_t))$$
$$d\theta_v \leftarrow d\theta_v + \frac{\partial (R - V_{\theta'_v}(s_t))^2}{\partial \theta'_v}$$

**[[02-模块/Actor-Critic/A2C|A2C]]（同步版本）**：所有 worker 同步收集数据，然后一次性更新。实践中 [[02-模块/Actor-Critic/A2C|A2C]] 与 [[02-模块/Actor-Critic/A3C|A3C]] 效果相当，且实现更简单。

### 优缺点

**优点**：
- 无需[[01-原子/经验回放|经验回放]]，内存效率高
- 并行化大幅加速训练（[[02-模块/Actor-Critic/A3C|A3C]] 在 Atari 上用单台多核 CPU 就超过 [[02-模块/Value-Based/DQN|DQN]] 用 GPU 的效果）
- 支持 on-policy 方法（如 Actor-Critic）
- [[02-模块/Actor-Critic/A2C|A2C]] 的同步版本特别适合 GPU 并行

**缺点**：
- [[02-模块/Actor-Critic/A3C|A3C]] 的异步更新可能导致参数过时（stale gradients）
- 多 worker 的负载均衡需要工程优化
- 在单 GPU 场景下 [[02-模块/Actor-Critic/A2C|A2C]] 通常已经足够

### 演化位置

[[02-模块/Actor-Critic/A3C|A3C]]/[[02-模块/Actor-Critic/A2C|A2C]] 证明了 Actor-Critic 可以高效并行化。这个思想直接影响了后续所有大规模 RL 系统——包括 [[02-模块/Policy-Based/PPO|PPO]] 的分布式实现、DeepSeek-R1 的训练系统、[[02-模块/Policy-Based/DAPO|DAPO]] 的 verl 框架。

---

## 九、[[02-模块/Actor-Critic/DDPG|DDPG]]：连续控制的深度确定性[[01-原子/策略梯度|策略梯度]]

### 来源与动机

- **论文**：Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D. (2016). "Continuous control with deep reinforcement learning." *ICLR 2016*. arXiv: 1509.02971.
- **解决的问题**：[[02-模块/Value-Based/DQN|DQN]] 只能处理离散动作空间（取 max over actions）。连续控制任务（机器人关节角度、力矩等）的动作空间是连续且高维的，无法枚举。

### 核心创新

> 把 [[02-模块/Value-Based/DQN|DQN]] 的 Q-learning 思想和确定性[[01-原子/策略梯度|策略梯度]]结合——Actor 输出确定性动作（不是概率分布），Critic 用 Q 函数评估。

### 关键公式

**确定性[[01-原子/策略梯度|策略梯度]]**（Silver et al., 2014）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\beta} [\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q(s,a)|_{a=\mu_\theta(s)}]$$

**[[02-模块/Actor-Critic/DDPG|DDPG]] 的两个关键技术**：
1. **Target Networks**：缓慢更新的 Actor/Critic [[01-原子/目标网络|目标网络]]，稳定训练
   $$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$
2. **[[01-原子/经验回放|经验回放]]**：存储 $(s, a, r, s')$ 转换，随机采样训练

### 优缺点

**优点**：
- 首次将深度 RL 成功应用于连续控制
- 在 20+ 个物理仿真任务上表现良好
- 可以从原始像素端到端学习

**缺点**：
- **Q 值过估计**：单个 Critic 倾向于高估 Q 值
- **对超参数极敏感**：学习率、target network 更新速率、噪声大小等
- 训练不稳定，不同随机种子结果差异大
- 只适用于确定性策略（不能直接学随机策略）

### 演化位置

[[02-模块/Actor-Critic/DDPG|DDPG]] 是连续控制领域的「[[02-模块/Value-Based/DQN|DQN]]」。它直接催生了 [[02-模块/Actor-Critic/TD3|TD3]]（修复过估计）和 [[02-模块/Actor-Critic/SAC|SAC]]（[[01-原子/最大熵原理|最大熵]]框架）。

---

## 十、[[02-模块/Actor-Critic/TD3|TD3]]：修复 [[02-模块/Actor-Critic/DDPG|DDPG]] 的三大缺陷

### 来源与动机

- **论文**：Fujimoto, S., van Hoof, H., Meger, D. (2018). "Addressing function approximation error in actor-critic methods." *ICML 2018*. arXiv: 1802.09477.
- **解决的问题**：[[02-模块/Actor-Critic/DDPG|DDPG]] 的 Q 值过估计导致策略学习被误导——Critic 说"这个动作很好"，但实际上没那么好，Actor 就朝着错误方向优化。

### 核心创新

> 三个关键技术：(1) Twin Critics（双 Q 网络取最小值）(2) Delayed Actor Update（延迟 Actor 更新）(3) Target Policy Smoothing（目标策略加噪平滑）。

### 关键公式

**Twin Critics**：
$$y = r + \gamma \min_{i=1,2} Q_{\phi'_i}(s', \pi_{\theta'}(s'))$$

取两个 Critic 的较小值，限制过估计。

**Target Policy Smoothing**：
$$\tilde{a}' = \pi_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

在目标动作上加噪声，防止 Critic 利用 Q 函数的尖锐峰值。

**Delayed Actor Update**：
Actor 每 $d$ 步更新一次（通常 $d=2$），让 Critic 先收敛得更准确。

### 优缺点

**优点**：
- 显著减少 Q 值过估计
- 训练比 [[02-模块/Actor-Critic/DDPG|DDPG]] 稳定得多
- 在 MuJoCo 等基准上全面超越 [[02-模块/Actor-Critic/DDPG|DDPG]]

**缺点**：
- 仍然是 on-policy 变体，样本效率不如 off-policy
- 不解决确定性策略的探索问题
- 对噪声超参数仍有一定敏感性

### 演化位置

[[02-模块/Actor-Critic/TD3|TD3]] 是 [[02-模块/Actor-Critic/DDPG|DDPG]] 的「bug-fix 版本」。它证明了 Q 值过估计是 actor-critic 的核心问题，并给出了实用解决方案。[[02-模块/Actor-Critic/SAC|SAC]] 在此基础上引入了[[01-原子/最大熵原理|最大熵]]框架。

---

## 十一、[[02-模块/Actor-Critic/SAC|SAC]]：[[01-原子/最大熵原理|最大熵]]强化学习

### 来源与动机

- **论文（v1）**：Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML 2018*. arXiv: 1801.01290.
- **论文（v2, 自动调温）**：Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., Levine, S. (2018). "Soft actor-critic algorithms and applications." arXiv: 1812.05905.
- **解决的问题**：[[02-模块/Actor-Critic/DDPG|DDPG]]/[[02-模块/Actor-Critic/TD3|TD3]] 虽然解决了连续控制问题，但 (1) 样本效率低（on-policy 或 near-on-policy），(2) 对超参数敏感，(3) 探索不足（确定性策略 + 高斯噪声）。

### 核心创新

> 在最大化回报的同时最大化策略熵——"完成任务的同时尽量随机行动"，自动温度调节 $\alpha$ 让探索-利用自动平衡。

### 关键公式

**[[01-原子/最大熵原理|最大熵]]目标**：
$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

其中 $\mathcal{H}(\pi(\cdot|s_t)) = -\sum_a \pi(a|s_t) \log \pi(a|s_t)$ 是策略熵，$\alpha$ 是温度参数。

**Soft Q 函数**（[[01-原子/Bellman方程|Bellman 方程]]）：
$$Q_\text{soft}(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p} [V_\text{soft}(s')]$$
$$V_\text{soft}(s) = \alpha \log \sum_a \exp\left(\frac{1}{\alpha} Q_\text{soft}(s,a)\right)$$

**自动温度调节（v2）**：
$$\min_\alpha \mathbb{E}_{a \sim \pi_\theta} [-\alpha \log \pi_\theta(a|s) - \alpha \bar{\mathcal{H}}]$$

其中 $\bar{\mathcal{H}}$ 是目标熵（通常设为 $-|\mathcal{A}|$）。这自动调节 $\alpha$：策略太确定时 $\alpha$ 增大（鼓励探索），太随机时 $\alpha$ 减小（专注利用）。

### 为什么[[01-原子/最大熵原理|最大熵]] + 自动调温重要？

1. **探索-利用自动平衡**：$\alpha$ 大 → 策略接近均匀分布 → 广泛探索；$\alpha$ 小 → 策略集中 → 精确利用
2. **多模态学习**：[[01-原子/最大熵原理|最大熵]]框架自然支持多模态策略（多个同样好的动作都可以有高概率）
3. **鲁棒性**：高熵策略对扰动更鲁棒（不依赖某个精确动作）
4. **无需手工调 $\alpha$**：自动调温消除了最关键的超参数

### 优缺点

**优点**：
- 样本效率高（off-policy，使用[[01-原子/经验回放|经验回放]]）
- 训练极其稳定（不同种子结果一致）
- 自动温度调节消除了关键超参数
- 在连续控制基准上全面领先（2018 年 SOTA）
- 理论上等价于最小化 KL 散度到最优策略

**缺点**：
- 只适用于连续动作空间（离散版本需要额外修改）
- [[01-原子/最大熵原理|最大熵]]目标可能不适合所有任务（某些任务需要确定性策略）
- 计算量比 [[02-模块/Policy-Based/PPO|PPO]] 大（需要 Twin Critics + Actor + 温度参数）

### 演化位置

[[02-模块/Actor-Critic/SAC|SAC]] 是连续控制领域的「终极形态」。它融合了 [[02-模块/Actor-Critic/DDPG|DDPG]] 的 actor-critic 结构、[[02-模块/Actor-Critic/TD3|TD3]] 的双 Critic 技术、以及[[01-原子/最大熵原理|最大熵]]框架的理论优雅。在机器人控制领域，[[02-模块/Actor-Critic/SAC|SAC]] 至今仍是首选方法。

---

## 十二、[[02-模块/Reward-Model/DPO|DPO]]：跳过奖励模型的直接偏好优化

### 来源与动机

- **论文**：Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023). "Direct preference optimization: Your language model is secretly a reward model." *NeurIPS 2023*. arXiv: 2305.18290.
- **解决的问题**：RLHF（Reinforcement Learning from Human Feedback）流程复杂且不稳定——先训练奖励模型，再用 [[02-模块/Policy-Based/PPO|PPO]] 优化。两步都有各自的困难：奖励模型可能不准确，[[02-模块/Policy-Based/PPO|PPO]] 可能不稳定。

### 核心创新

> 数学推导出：奖励模型可以被策略本身参数化——"你的语言模型本身就是一个奖励模型"，直接用分类损失优化，完全跳过奖励模型和 RL。

### 关键公式

**RLHF 的标准目标**：
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x,y)] - \beta D_\text{KL}(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x))$$

**[[02-模块/Reward-Model/DPO|DPO]] 的闭式解（关键推导）**：

最优策略可以用参考策略和奖励函数表达：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

反过来，奖励函数可以用策略表达：
$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

**[[02-模块/Reward-Model/DPO|DPO]] 损失函数**（代入 [[01-原子/Bradley-Terry模型|Bradley-Terry]] 偏好模型，$Z(x)$ 项消去）：

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

其中 $y_w$ 是人类偏好的回复（winner），$y_l$ 是非偏好的回复（loser）。

### 优缺点

**优点**：
- **极简**：只需要一个分类损失，不需要奖励模型、不需要 RL 训练
- **稳定**：没有 [[02-模块/Policy-Based/PPO|PPO]] 的不稳定性，训练像监督学习一样
- **高效**：计算量只有 [[02-模块/Policy-Based/PPO|PPO]]-RLHF 的几分之一
- **效果好**：在情感控制、摘要、对话等任务上匹配或超越 [[02-模块/Policy-Based/PPO|PPO]]-RLHF

**缺点**：
- 是 off-line 方法——只能从静态偏好数据学习，不能在线探索
- 对数据质量敏感——偏好标注错误会直接影响模型
- 无法利用在线反馈（后续工作如 online [[02-模块/Reward-Model/DPO|DPO]] 试图解决）
- 在某些场景下（如需要复杂推理的任务）不如 [[02-模块/Policy-Based/GRPO|GRPO]]/[[02-模块/Policy-Based/PPO|PPO]]

### 演化位置

[[02-模块/Reward-Model/DPO|DPO]] 是 LLM 对齐领域的「范式转移」。它证明了不需要 RL 也可以做偏好对齐。[[02-模块/Reward-Model/DPO|DPO]] 催生了大量变体（IPO、KTO、SimPO、ORPO 等），成为与 [[02-模块/Policy-Based/PPO|PPO]]-RLHF 并列的两大对齐范式之一。

---

## 十三、[[02-模块/Policy-Based/GRPO|GRPO]]：不需要 Critic 的组相对策略优化

### 来源与动机

- **论文**：Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., Guo, D. (2024). "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." arXiv: 2402.03300.
- **机构**：DeepSeek（深度求索）
- **解决的问题**：[[02-模块/Policy-Based/PPO|PPO]] 需要一个 Critic（值函数）网络来估计 Advantage。对于 LLM 来说，Critic 网络和 Actor 网络一样大（都是 LLM），内存和计算开销翻倍。而且 [[01-原子/GAE|GAE]] 需要 Critic 准确估计 $V(s)$，这在 LLM 场景极难。

### 核心创新

> 去掉 Critic，用"组内相对排名"估计 Advantage——对同一个 prompt 生成一组回复，用组内均值和标准差归一化奖励作为 Advantage。

### 关键公式

**[[02-模块/Policy-Based/GRPO|GRPO]] Advantage 估计**：

对问题 $q$，行为策略 $\pi_{\theta_\text{old}}$ 生成 $G$ 个回复 $\{o_i\}_{i=1}^G$，第 $i$ 个回复的 Advantage：

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

**[[02-模块/Policy-Based/GRPO|GRPO]] 目标函数**：

$$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left(r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}\right) - \beta D_\text{KL}(\pi_\theta \| \pi_\text{ref}) \right) \right]$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}|q, o_{i,<t})}$。

### 组归一化如何替代 Critic？

**PPO 的 Advantage**：$\hat{A}_t = R_t - V(s_t)$，需要一个 Critic 估计 $V(s_t)$。

**GRPO 的 Advantage**：$\hat{A}_i = \frac{R_i - \bar{R}}{\sigma_R}$，只需要同一 prompt 下多个回复的奖励统计量。

**直觉**：当奖励是二值的（对/错，+1/-1），组均值 $\bar{R}$ 就是"这个 prompt 的通过率"，它自然起到了 baseline 的作用。$\hat{A}_i > 0$ 意味着"这个回复比组内平均好"。

**KL 惩罚的作用**：[[02-模块/Policy-Based/GRPO|GRPO]] 保留了 $-\beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ 项，防止策略偏离参考模型太远。这在 RLHF 场景很重要，但在纯数学推理 RL 中（如 [[02-模块/Policy-Based/DAPO|DAPO]]），可以移除。

### 优缺点

**优点**：
- **不需要 Critic 网络**：内存和计算节省约 50%（省掉一个 LLM 大小的值函数网络）
- 实现比 [[02-模块/Policy-Based/PPO|PPO]] 简单得多
- 在数学推理任务上表现优异（DeepSeekMath 7B 在 MATH 上达到 51.7%）
- 被 DeepSeek-R1 采用，验证了大规模 RL 的有效性

**缺点**：
- 需要为每个 prompt 生成多个回复（$G$ 通常 8-64），推理成本高
- 当组内所有回复奖励相同时（全对或全错），Advantage 为零 → 梯度为零 → 浪费样本
- 组归一化的 Advantage 估计比 [[01-原子/GAE|GAE]] 方差更大
- 存在熵崩溃问题（[[02-模块/Policy-Based/DAPO|DAPO]] 论文指出并解决）

### 演化位置

[[02-模块/Policy-Based/GRPO|GRPO]] 是 [[02-模块/Policy-Based/PPO|PPO]] 在 LLM RL 场景的「极简变体」。它去掉了最昂贵的 Critic 组件，用组内统计替代值函数估计。[[02-模块/Policy-Based/GRPO|GRPO]] 成为 DeepSeek-R1 的核心算法，也是 [[02-模块/Policy-Based/DAPO|DAPO]] 的直接前身。

---

## 十四、[[02-模块/Policy-Based/DAPO|DAPO]]：大规模 LLM RL 的开源系统

### 来源与动机

- **论文**：Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., et al. (2025). "[[02-模块/Policy-Based/DAPO|DAPO]]: An open-source LLM reinforcement learning system at scale." arXiv: 2503.14476. ByteDance Seed & Tsinghua AIR.
- **解决的问题**：用 [[02-模块/Policy-Based/GRPO|GRPO]] 复现 DeepSeek-R1 的效果时，初始 [[02-模块/Policy-Based/GRPO|GRPO]] 实验只得到 AIME 2024 上 30 分（DeepSeek-R1-Zero-Qwen-32B 为 47 分）。深入分析发现 [[02-模块/Policy-Based/GRPO|GRPO]] 存在四个关键问题：熵崩溃、奖励噪声、训练不稳定、样本利用率低。

### 核心创新

> 四个关键技术修复 [[02-模块/Policy-Based/GRPO|GRPO]] 在大规模 LLM RL 中的缺陷：(1) Clip-Higher 解耦裁剪上下界 (2) 动态采样过滤零梯度样本 (3) Token 级别损失计算 (4) 超长奖励塑形。

### 关键公式与技术

#### 技术 1：Clip-Higher（解耦裁剪）

[[02-模块/Policy-Based/GRPO|GRPO]]/[[02-模块/Policy-Based/PPO|PPO]] 的 clip 范围是对称的：$[1-\epsilon, 1+\epsilon]$。[[02-模块/Policy-Based/DAPO|DAPO]] 发现上界 clip 过紧会限制探索：

**问题**：当 $\epsilon=0.2$，一个低概率 token（$\pi=0.01$）的概率上限只能增加到 $0.012$，而高概率 token（$\pi=0.9$）可以增到 $1.08$（实际无限制）。低概率的"探索性"token 被锁死了。

**解法**：解耦上下界：

$$\text{clip}(r_{i,t}(\theta), 1-\epsilon_\text{low}, 1+\epsilon_\text{high})$$

其中 $\epsilon_\text{high} > \epsilon_\text{low}$（例如 $\epsilon_\text{low}=0.2, \epsilon_\text{high}=0.28$），允许低概率 token 有更大的上升空间。

#### 技术 2：Dynamic Sampling（动态采样）

**问题**：随着训练进行，越来越多的 prompt 全对（accuracy=1）或全错（accuracy=0），组内 Advantage 全为零，梯度为零，样本被浪费。

**解法**：过采样并过滤掉 accuracy=0 和 accuracy=1 的 prompt，只保留有有效梯度的样本：

$$\text{s.t.} \quad 0 < |\{o_i \mid \texttt{is\_equivalent}(a, o_i)\}| < G$$

#### 技术 3：Token-Level Policy Gradient Loss

**问题**：GRPO 用 sample-level loss（先按 sample 取平均，再跨 sample 取平均），长回复中的 token 权重被稀释。

**解法**：用 token-level loss，按总 token 数归一化：

$$\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min(\cdot)$$

替代 GRPO 的 $\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$。

#### 技术 4：Overlong Reward Shaping

**问题**：超长回复被截断后赋予惩罚奖励 $-1$，但一个"推理过程正确但太长"的回复被惩罚会引入噪声。

**解法**：对被截断的样本做特殊处理（过滤或软惩罚），避免错误信号。

#### DAPO 完整目标

$$\mathcal{J}_\text{DAPO}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon_\text{low}, 1+\epsilon_\text{high}) \hat{A}_{i,t}\right) \right]$$

$$\text{s.t.} \quad 0 < |\{o_i \mid \texttt{is\_equivalent}(a, o_i)\}| < G$$

**无 KL 惩罚项**——DAPO 认为在数学推理 RL 中，策略可以大幅偏离预训练模型，KL 约束不必要。

### 优缺点

**优点**：
- 在 AIME 2024 上达到 50 分（超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分），且只用 50% 训练步数
- 完全开源——代码、数据、模型全部公开
- 揭示了 GRPO 在大规模 RL 中的四个关键问题
- 基于 verl 框架，工程可复现

**缺点**：
- 动态采样增加了推理开销（需要过采样）
- Clip-Higher 的 $\epsilon_\text{high}$ 需要额外调节
- 目前主要在数学推理上验证，其他任务（代码、对话）的泛化性待验证

### 演化位置

DAPO 是 GRPO 的「工程化升级版」，是当前（2025 年）LLM RL 的开源 SOTA。它证明了在大规模 LLM RL 中，细节（clip 策略、采样策略、损失计算方式、奖励塑形）决定成败。

---

## 十五、DCPO 及其他 2025-2026 最新进展

### DCPO（Dynamic Clipping Policy Optimization）

- **论文**：Yang, S., Dou, C., Guo, P., et al. (2025). arXiv: 2509.02333.
- **核心改进**：
  1. **动态 clip 边界**：根据 token 的先验概率自适应调整 clip 范围（不是 DAPO 的固定解耦，而是动态计算）
  2. **平滑 Advantage 标准化**：跨累积训练步标准化奖励（不是单 batch 内标准化）
- **结果**：在 AIME24 上超过 DAPO（46.7 vs 36.7 Avg@1，Qwen2.5-Math-7B），训练效率翻倍

### 2025-2026 LLM RL 新趋势

1. **RLVR（RL from Verifiable Rewards）**：用可验证的奖励（数学题答案对错、代码测试通过率）替代人类偏好和奖励模型。GRPO/DAPO/DCPO 都属于这一范式。

2. **Scaf-GRPO（2025/2026，ICLR 2026）**：解决 GRPO 的"学习悬崖"问题——当模型完全不会做某题时，所有回复都错，Advantage 全为零。Scaf-GRPO 在模型独立学习停滞时注入分层提示（从抽象概念到具体步骤）。

3. **N-GRPO（2026，ACL 2026 Findings）**：用语义邻居混合（Semantic Neighbor Mixing）替代 token 级采样，在 embedding 层面注入多样性，解决 GRPO 的探索不足问题。

4. **PPO-BR（2025）**：双信号（熵 + 奖励）自适应信赖域——探索阶段用熵信号扩张 clip 范围，收敛阶段用奖励信号收缩 clip 范围。

---

## 三条主线深度分析

### 主线一：REINFORCE → Actor-Critic → TRPO → PPO（通用策略优化）

| 方法 | 年份 | 核心突破 | 解决的前一方法缺陷 |
|------|------|----------|-------------------|
| REINFORCE | 1992 | 直接对策略求梯度 | 值函数方法不能处理连续动作 |
| Baseline | ~1999 | 减去 $V(s)$ 降方差 | REINFORCE 方差太大 |
| Actor-Critic | 1983/2000 | Critic 提供即时反馈 | REINFORCE 需等整条轨迹 |
| Natural PG | 2002 | Fisher 信息矩阵校正方向 | 标准梯度在参数空间不等价于策略空间 |
| TRPO | 2015 | KL 约束保证单调改进 | 策略更新步长太大会崩溃 |
| A3C/A2C | 2016 | 并行化加速 | 经验回放不支持 on-policy |
| PPO | 2017 | Clip 替代 KL，简化实现 | TRPO 实现太复杂 |

**演化逻辑**：每一步都在解决前一步的核心瓶颈——方差（Baseline）→ 样本效率（AC）→ 方向正确性（Natural PG）→ 步长安全性（TRPO）→ 工程简洁性（PPO）。

### 主线二：DDPG → TD3 → SAC（连续控制）

| 方法 | 年份 | 核心突破 | 解决的前一方法缺陷 |
|------|------|----------|-------------------|
| DDPG | 2016 | 确定性策略梯度 + 深度网络 | Q-learning 不能处理连续动作 |
| TD3 | 2018 | 双 Critic + 延迟更新 + 目标平滑 | DDPG 的 Q 值过估计 |
| SAC | 2018 | 最大熵 + 自动温度调节 | TD3 的探索不足和超参敏感 |

**演化逻辑**：DDPG 开了连续控制深度 RL 的先河 → TD3 修复了 Q 值过估计的工程 bug → SAC 用最大熵框架统一了探索和利用。SAC 的自动温度调节是"探索-利用"问题的优雅解法。

### 主线三：PPO → GRPO → DAPO → DPO（LLM 对齐）

| 方法 | 年份 | 核心突破 | 解决的前一方法缺陷 |
|------|------|----------|-------------------|
| PPO-RLHF | 2022 | PPO + 奖励模型 + KL 约束 | 直接微调无法对齐人类偏好 |
| DPO | 2023 | 跳过奖励模型，直接分类优化 | PPO-RLHF 太复杂且不稳定 |
| GRPO | 2024 | 去掉 Critic，组内相对 Advantage | PPO 的 Critic 在 LLM 场景太贵 |
| DAPO | 2025 | 四个工程修复 GRPO 的缺陷 | GRPO 的熵崩溃、零梯度、样本浪费 |
| DCPO | 2025 | 动态 clip + 跨步标准化 | DAPO/GRPO 的固定 clip 边界 |

**演化逻辑**：PPO-RLHF 太贵太复杂 → DPO 跳过 RL（但牺牲了在线探索能力）→ GRPO 简化 PPO 但保留 RL（去 Critic）→ DAPO 修复 GRPO 的工程缺陷 → DCPO 进一步优化 clip 策略。

**两条路线的对比**：
- **在线 RL 路线**（PPO → GRPO → DAPO）：需要模型在线生成回复，可以探索新行为，适合推理能力强化
- **离线偏好路线**（DPO 系列）：从静态偏好数据学习，更简单稳定，适合行为对齐

---

## 面试核心问答

### Q1: 为什么 TRPO 用 KL 约束而 PPO 用 Clip？

**答**：TRPO 的理论证明要求策略更新在 KL 球内，保证单调改进。但 KL 约束需要计算 Fisher 矩阵（二阶信息），实现复杂。PPO 观察到：clip 是一种更简单的"近似信赖域"——它把概率比限制在 $[1-\epsilon, 1+\epsilon]$，效果上近似于 KL 约束（限制了策略变化的幅度），但只需要一行代码。PPO 牺牲了严格的理论保证，换取了极大的工程简洁性。实践中 PPO 的效果与 TRPO 相当甚至更好（因为 PPO 可以多次更新同一批数据，样本效率更高）。

### Q2: SAC 的最大熵和自动温度调节为什么重要？

**答**：
- **最大熵**：传统 RL 只最大化回报，策略可能变得完全确定性。SAC 额外最大化熵，策略在完成任务的同时保持随机性。这带来三个好处：(1) 更好的探索 (2) 多模态学习（多个好动作都有概率）(3) 鲁棒性（不依赖精确动作）。
- **自动温度 $\alpha$**：$\alpha$ 控制"探索 vs 利用"的平衡。手动调 $\alpha$ 很困难——太大会过度随机，太小会探索不足。SAC v2 通过约束优化自动调节：$\min_\alpha \mathbb{E}[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}]$。当策略熵低于目标 $\bar{\mathcal{H}}$ 时 $\alpha$ 自动增大，高于时减小。这消除了最关键的超参数。

### Q3: GRPO 的组归一化如何替代 Critic？

**答**：PPO 用 Critic 估计 $V(s_t)$，Advantage = $R_t - V(s_t)$。GRPO 不用 Critic，而是对同一 prompt 生成 $G$ 个回复，Advantage = $(R_i - \bar{R}) / \sigma_R$。这里的 $\bar{R}$ 扮演了 baseline 的角色——它代表了"这个 prompt 的平均难度"。直觉上：如果 64 个回复中 40 个对了，$\bar{R} \approx 0.625$，一个正确的回复 Advantage = $(1 - 0.625) / \sigma > 0$。这不需要额外的值函数网络，节省了约 50% 的模型参数和计算。但代价是需要生成多个回复（推理成本高），且 Advantage 估计的方差比 [[01-原子/GAE|GAE]] 大。

### Q4: [[02-模块/Reward-Model/DPO|DPO]] 和 [[02-模块/Policy-Based/GRPO|GRPO]]/[[02-模块/Policy-Based/DAPO|DAPO]] 的本质区别？

**答**：
- **[[02-模块/Reward-Model/DPO|DPO]]**：离线（offline），从静态偏好数据学习，不需要模型在线生成回复。把 RL 问题转化为分类问题。
- **[[02-模块/Policy-Based/GRPO|GRPO]]/[[02-模块/Policy-Based/DAPO|DAPO]]**：在线（online），需要模型实时生成回复并用可验证奖励评分。保留了 RL 的探索能力。
- **选择**：如果目标是"行为对齐"（让模型说人话、不有害），[[02-模块/Reward-Model/DPO|DPO]] 足够。如果目标是"能力强化"（提升推理、解题），需要在线 RL（[[02-模块/Policy-Based/GRPO|GRPO]]/[[02-模块/Policy-Based/DAPO|DAPO]]）。

---

## 参考文献

1. Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*.
2. Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (1999). "Policy gradient methods for reinforcement learning with function approximation." *NIPS*.
3. Barto, A. G., Sutton, R. S., Anderson, C. W. (1983). "Neuronlike adaptive elements." *IEEE TSMC*.
4. Konda, V. R. & Tsitsiklis, J. N. (2000). "Actor-critic algorithms." *NIPS*.
5. Kakade, S. M. (2002). "A natural policy gradient." *NIPS*.
6. Schulman, J. et al. (2015). "Trust region policy optimization." *ICML*. arXiv: 1502.05477.
7. Schulman, J. et al. (2017). "Proximal policy optimization algorithms." arXiv: 1707.06347.
8. Mnih, V. et al. (2016). "Asynchronous methods for deep reinforcement learning." *ICML*. arXiv: 1602.01783.
9. Lillicrap, T. P. et al. (2016). "Continuous control with deep reinforcement learning." *ICLR*. arXiv: 1509.02971.
10. Fujimoto, S. et al. (2018). "Addressing function approximation error in actor-critic methods." *ICML*. arXiv: 1802.09477.
11. Haarnoja, T. et al. (2018). "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning." *ICML*. arXiv: 1801.01290.
12. Haarnoja, T. et al. (2018). "Soft actor-critic algorithms and applications." arXiv: 1812.05905.
13. Rafailov, R. et al. (2023). "Direct preference optimization: Your language model is secretly a reward model." *NeurIPS*. arXiv: 2305.18290.
14. Shao, Z. et al. (2024). "DeepSeekMath: Pushing the limits of mathematical reasoning." arXiv: 2402.03300.
15. Yu, Q. et al. (2025). "[[02-模块/Policy-Based/DAPO|DAPO]]: An open-source LLM reinforcement learning system at scale." arXiv: 2503.14476.
16. Yang, S. et al. (2025). "DCPO: Dynamic clipping policy optimization." arXiv: 2509.02333.
17. Zhang, X. et al. (2025). "Scaf-[[02-模块/Policy-Based/GRPO|GRPO]]: Scaffolded group relative policy optimization." arXiv: 2510.19807. *ICLR 2026*.
18. Zhu, X. et al. (2026). "N-[[02-模块/Policy-Based/GRPO|GRPO]]: Embedding-level neighbor mixing for enhanced policy optimization." arXiv: 2606.10768. *ACL 2026 Findings*.
19. Rahman, B. (2025). "[[02-模块/Policy-Based/PPO|PPO]]-BR: Dual-signal entropy-reward adaptation for trust region policy optimization." arXiv: 2505.17714.
