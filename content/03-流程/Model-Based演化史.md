# 基于模型的强化学习 (Model-Based RL) 完整演化史

> **核心主线**:MBRL 的历史就是人类与「模型误差累积」这一根本挑战斗争的历史。三条主线并行演化,各自用不同策略对抗误差:Dyna-[[02-模块/Model-Based/PETS|PETS]]-[[02-模块/Model-Based/MBPO|MBPO]] 系(短 rollout 截断误差)、World Models-Dreamer 系(潜空间压缩误差)、MCTS-AlphaGo-MuZero 系(搜索+价值等价绕过误差)。2024-2026 年,视频生成模型作为世界模型成为最新趋势。

---

## 目录

1. [总论:为什么需要模型?](#1-总论为什么需要模型)
2. [根本挑战:模型误差累积](#2-根本挑战模型误差累积)
3. [动态规划 — Bellman (1950s)](#3-动态规划--bellman-1950s)
4. [[[02-模块/Model-Based/Dyna-Q|Dyna-Q]] — Sutton (1991)](#4-dyna-q--sutton-1991)
5. [MCTS / UCT — Kocsis & Szepesvári (2006)](#5-mcts--uct--kocsis--szepesvári-2006)
6. [World Models — Ha & Schmidhuber (2018)](#6-world-models--ha--schmidhuber-2018)
7. [[[02-模块/Model-Based/PETS|PETS]] — Chua et al. (2018)](#7-pets--chua-et-al-2018)
8. [[[02-模块/Model-Based/MBPO|MBPO]] — Janner et al. (2019)](#8-mbpo--janner-et-al-2019)
9. [Dreamer v1 — Hafner et al. (2020)](#9-dreamer-v1--hafner-et-al-2020)
10. [MuZero — Schrittwieser et al. (2020)](#10-muzero--schrittwieser-et-al-2020)
11. [Dreamer v2 — Hafner et al. (2021)](#11-dreamer-v2--hafner-et-al-2021)
12. [IRIS — Micheli, Alonso & Fleuret (2023)](#12-iris--micheli-alonso--fleuret-2023)
13. [Dreamer v3 — Hafner et al. (2023)](#13-dreamer-v3--hafner-et-al-2023)
14. [TD-MPC2 — Hansen et al. (2024)](#14-td-mpc2--hansen-et-al-2024)
15. [Genie — Bruce et al. / DeepMind (2024)](#15-genie--bruce-et-al--deepmind-2024)
16. [GameNGen — Valevski et al. / Google (2024)](#16-gamengen--valevski-et-al--google-2024)
17. [视频生成作为世界模型:Genie 2 / UniSim / Sora 类 (2024-2026)](#17-视频生成作为世界模型genie-2--unisim--sora-类-2024-2026)
18. [2025-2026 最新进展](#18-2025-2026-最新进展)
19. [三大主线对比总结](#19-三大主线对比总结)
20. [模型误差对抗策略全景](#20-模型误差对抗策略全景)
21. [参考文献](#21-参考文献)

---

## 1. 总论:为什么需要模型?

**一句话**:Model-Free RL 靠试错,Model-Based RL 靠「想象」——先用模型学会世界怎么运转,然后在想象中规划最优行动。

强化学习的核心困境是**样本效率**(sample efficiency)。Model-Free 方法(如 [[02-模块/Value-Based/DQN|DQN]]、[[02-模块/Policy-Based/PPO|PPO]]、[[02-模块/Actor-Critic/SAC|SAC]])需要数百万甚至数十亿次环境交互才能收敛,这在真实机器人、自动驾驶等场景中不可接受。

**MBRL 的核心思想**:
1. 从已有数据中学一个**环境模型**(dynamics model): $\hat{s}_{t+1} = f(s_t, a_t)$
2. 用模型生成**想象轨迹**(imagined rollouts)来替代真实交互
3. 在想象轨迹上训练策略,从而大幅减少真实环境交互

**核心权衡**:模型越好 → 想象越可靠 → 策略越好;但模型永远有误差 → 误差在 rollout 中累积 → 策略可能利用模型错误而非真实规律。

---

## 2. 根本挑战:模型误差累积

**大白话**:你用一个不完美的水晶球预测未来,预测越远,偏差越大。如果你基于这个预测做决策,偏差会被放大,最终完全偏离现实。

**数学表达**:设真实动力学为 $s_{t+1} = f^*(s_t, a_t)$,学到的模型为 $\hat{f}$,单步误差为 $\epsilon = \|\hat{f} - f^*\|$。在 $H$ 步 rollout 中:

$$\text{累积误差} \leq \sum_{t=0}^{H-1} \gamma^t \cdot \epsilon_t \approx \frac{\epsilon}{1-\gamma} \cdot H \quad (\text{最坏情况线性增长})$$

更严重的是**分布偏移**(distributional shift):策略在想象轨迹上优化后,会倾向于探索模型训练数据未覆盖的区域,导致模型在这些区域的误差更大,形成恶性循环。

**三条对抗路线**:

| 路线 | 策略 | 代表方法 |
|------|------|----------|
| 截断误差传播 | 只用短 rollout,从真实数据出发 | [[02-模块/Model-Based/Dyna-Q|Dyna-Q]] → [[02-模块/Model-Based/PETS|PETS]] → [[02-模块/Model-Based/MBPO|MBPO]] |
| 压缩状态空间 | 在潜空间中想象,误差在低维空间中更可控 | World Models → Dreamer → IRIS |
| 绕过显式动力学 | 不预测完整下一状态,只预测价值/策略相关量 | MCTS → AlphaGo → MuZero |

---

## 3. 动态规划 — Bellman (1950s)

### 来源与动机

Richard Bellman 在 1950 年代提出动态规划(Dynamic Programming),首次系统性地给出了「已知环境模型时如何求最优策略」的数学框架。这是 MBRL 的理论鼻祖——当转移概率 $P(s'|s,a)$ 和奖励函数 $R(s,a)$ 完全已知时,可以精确求解。

### 核心创新

**[[01-原子/Bellman方程|Bellman 方程]]**:将最优决策问题分解为递归的子问题:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$

### 关键方法

**值迭代 (Value Iteration)**:
$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$
反复迭代直到收敛。

**策略迭代 (Policy Iteration)**:
1. 策略评估:对当前策略 $\pi$ 求解 $V^\pi$
2. 策略改进:$\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]$
3. 重复直到策略不再改变

### 优缺点

- **优点**:理论上保证收敛到最优策略;数学优美、严格
- **缺点**:需要**完整已知**的模型(转移概率矩阵);状态空间大时计算不可行(维度灾难);无法处理连续状态空间
- **演化位置**:理论基石,但实际中模型通常未知 → 催生了两个方向:Model-Free(TD-learning, Q-learning)和 Model-Based(学模型 + 用模型规划)

---

## 4. [[02-模块/Model-Based/Dyna-Q|Dyna-Q]] — Sutton (1991)

### 来源与动机

Richard Sutton 在 1991 年的论文 "Dyna, an Integrated Architecture for Learning, Planning, and Reacting" 中提出。核心问题:当模型未知时,能否**一边学模型,一边用模型生成的数据加速学习**?

这是 MBRL 从理论走向实践的第一步——不再假设模型已知,而是从经验中学习一个近似模型,然后用模型生成模拟经验来补充真实经验。

### 核心创新

**Dyna 架构**:将学习(learning)、规划(planning)、行动(acting)集成在一个循环中:

```
真实环境交互 → 更新模型 → 用模型生成模拟经验 → 更新 Q 值 → 选择行动
                      ↑                                    ↓
                      └────── planning (n 步模拟) ──────────┘
```

每次真实交互后,从模型中采样 $n$ 步模拟经验(称为 **planning steps**),用这些模拟经验也来更新 Q 值。

### 关键公式/架构

1. **直接学习 (Direct RL)**:用真实 $(s, a, r, s')$ 更新 Q-table
2. **模型学习**:$Model(s,a) \leftarrow (r, s')$ (确定性环境中直接记录)
3. **规划**:随机选取之前见过的 $(s,a)$,从 $Model$ 中查询 $(r,s')$,用 $(s,a,r,s')$ 再做一次 Q-learning 更新

### 优缺点

- **优点**:比纯 Model-Free 的 Q-learning 样本效率高很多(每步真实交互额外做 $n$ 步规划);架构简洁优雅;首次证明了"学模型 + 用模型"的可行性
- **缺点**:用 tabular 模型(直接记忆),无法泛化到新状态;没有处理模型误差的机制——如果模型错了,Q 值会被模拟的错误经验带偏;只适用于离散小状态空间
- **演化位置**:MBRL 的开创性工作。Dyna 的"从真实数据出发做短 rollout"思想直接影响了后来的 [[02-模块/Model-Based/PETS|PETS]](2018)和 [[02-模块/Model-Based/MBPO|MBPO]](2019)。Dyna 没解决的模型误差问题,催生了后续几十年的研究

---

## 5. MCTS / UCT — Kocsis & Szepesvári (2006)

### 来源与动机

Levente Kocsis 和 Csaba Szepesvári 在 2006 年提出 UCT(Upper Confidence bounds applied to Trees)算法,将多臂赌博机的 UCB 策略扩展到树搜索。MCTS 解决的核心问题:在巨大状态空间中(如围棋, $10^{170}$ 个合法局面),如何不用完整模型就能做出好的决策?

### 核心创新

**蒙特卡洛树搜索 (MCTS)**:通过大量随机模拟(rollout)来评估局面好坏,用 UCB 公式平衡探索与利用:

$$UCT(s, a) = \bar{Q}(s,a) + C \sqrt{\frac{\ln N(s)}{N(s,a)}}$$

其中 $\bar{Q}(s,a)$ 是平均回报,$N(s)$ 是父节点访问次数,$N(s,a)$ 是边 $(s,a)$ 访问次数,$C$ 是探索常数。

**四个阶段循环**:
1. **Selection**:从根节点沿 UCT 值最大的路径向下
2. **Expansion**:扩展一个新节点
3. **Simulation**:从新节点随机模拟到终局
4. **Backpropagation**:将结果回传更新统计量

### 优缺点

- **优点**:不需要评估函数(只需终局结果); anytime 性质——随时可以停止并返回当前最优;天然处理巨大状态空间;理论上渐进收敛到最优解
- **缺点**:在稀疏奖励/长 horizon 任务中效率低(随机 rollout 很难到达好的终局);需要**环境模拟器**(access to simulator),不能纯从数据工作;计算量大
- **演化位置**:MCTS 是 AlphaGo(2016)的核心搜索组件。AlphaGo 用神经网络替代了 MCTS 的随机 rollout,用策略网络指导搜索方向、价值网络评估叶节点。这条路线最终在 MuZero(2020)中达到顶峰——既学模型又搜索,不需要真实环境规则

---

## 6. World Models — Ha & Schmidhuber (2018)

### 来源与动机

David Ha 和 Jurgen Schmidhuber 在 2018 年发表 "World Models"(arXiv:1803.10122)。核心思想:人类在脑中构建世界的内部模型,然后在"想象"中规划行动。能否让 AI 也这样做?

这是**潜空间想象 (latent imagination)** 路线的开创性工作。

### 核心创新

三组件架构:
1. **V (Vision)**:VAE 编码器,将高维观测压缩为低维潜变量 $z_t$
2. **M (Memory)**:MDN-RNN(Mixture Density Network + RNN),预测下一个潜变量的分布 $P(z_{t+1} | z_t, a_t, h_t)$,其中 $h_t$ 是 RNN 隐状态
3. **C (Controller)**:极简线性策略,输入为 $(z_t, h_t)$,输出动作 $a_t$。参数量仅约 600 个

**关键洞察**:策略在 VAE + RNN 构成的"梦境"(hallucinated dream)中训练,然后直接迁移到真实环境。

### 关键公式/架构

$$\text{VAE 损失: } \mathcal{L}_V = -\log P(x_t | z_t) + KL(q(z_t|x_t) \| P(z_t))$$
$$\text{RNN 损失: } \mathcal{L}_M = -\log P(z_{t+1} | z_t, a_t, h_t)$$
$$\text{Controller: } a_t = \tanh(W_c [z_t; h_t] + b_c)$$

策略参数 $W_c, b_c$ 通过 CMA-ES(进化策略)优化,目标是在梦境中最大化累积奖励。

### 优缺点

- **优点**:首次展示了在"梦境"中学习复杂行为的可行性;架构优雅、有生物学直觉;极大压缩了策略参数量
- **缺点**:RNN 的长期预测能力有限,误差在梦境中快速累积;VAE 的重建目标不一定学到对决策有用的表征;只在相对简单的环境(VizDoom、CarRacing)中验证;Controller 用进化策略优化,效率不高
- **演化位置**:开创了"VAE + 序列模型 + 潜空间想象"的范式。直接催生了 Dreamer 系列(Hafner et al. 2020-2023)。RSSM(Recurrent State-Space Model)就是对 V+M 的改进版

---

## 7. [[02-模块/Model-Based/PETS|PETS]] — Chua et al. (2018)

### 来源与动机

Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine 发表在 NeurIPS 2018(arXiv:1805.12114)。论文标题: "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"。

核心问题:之前的 Model-Based 方法要么用简单模型(如线性动力学,无法拟合复杂环境),要么用高容量模型(如神经网络,但模型偏差导致策略利用模型错误)。如何**在保持高容量模型表达力的同时,量化并利用模型不确定性来避免被模型偏差误导**?

### 核心创新

**概率集成 + 轨迹采样 (Probabilistic Ensembles with Trajectory Sampling)**:

1. **概率集成动力学模型**:训练 $K$ 个独立的神经网络(通常 $K=5$),每个预测下一步状态的**均值和方差**:
   $$\hat{s}_{t+1} \sim \mathcal{N}(\mu_{\theta_k}(s_t, a_t), \Sigma_{\theta_k}(s_t, a_t))$$

2. **轨迹采样 (Trajectory Sampling)**:规划时,从集成中**随机选取**一个模型来预测每一步,从而自然地传播模型不确定性:
   - 在集成分歧大的区域(模型不确定),轨迹会发散 → 控制器变得保守
   - 在集成一致的区域(模型确定),轨迹集中 → 控制器可以激进

3. **交叉熵方法 (CEM) 规划**:在想象轨迹上优化动作序列,用 CEM 迭代采样和筛选。

### 关键公式/架构

$$\mu(s_t, a_t) = \frac{1}{K} \sum_{k=1}^{K} \mu_{\theta_k}(s_t, a_t)$$
$$\sigma^2(s_t, a_t) = \frac{1}{K} \sum_{k=1}^{K} [\sigma^2_{\theta_k}(s_t, a_t) + \mu^2_{\theta_k}(s_t, a_t)] - \mu^2(s_t, a_t)$$

集成的分歧(方差)同时捕捉了**认知不确定性**(epistemic uncertainty,模型不知道的)和**偶然不确定性**(aleatoric uncertainty,环境本身的噪声)。

### 优缺点

- **优点**:样本效率极高——比 [[02-模块/Actor-Critic/SAC|SAC]] 少 8 倍、比 [[02-模块/Policy-Based/PPO|PPO]] 少 125 倍的样本达到相当性能;不确定性量化有效防止策略利用模型错误;架构简单,容易实现
- **缺点**:计算成本高(需要训练 $K$ 个网络,每步规划评估多条轨迹);只适用于连续控制的低维状态空间;CEM 规划在高维动作空间中效率下降;没有端到端训练——模型和策略分开优化
- **演化位置**:Dynasty 路线的关键里程碑。将 Dyna 的"学模型 + 用模型生成经验"升级为概率版本,用集成不确定性对抗模型误差。直接影响了 [[02-模块/Model-Based/MBPO|MBPO]](2019)的设计

---

## 8. [[02-模块/Model-Based/MBPO|MBPO]] — Janner et al. (2019)

### 来源与动机

Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine 发表在 NeurIPS 2019(arXiv:1906.08253)。论文标题: "When to Trust Your Model: Model-Based Policy Optimization"。

核心问题:[[02-模块/Model-Based/PETS|PETS]] 虽然样本效率高但计算成本大,而且长 rollout 中误差累积严重。**到底该多信任模型?rollout 该多长?**

### 核心创新

**理论分析 + 短 rollout 策略**:

1. **理论分析**:证明了策略改进的下界:
   $$\eta(\pi_{new}) \geq \eta(\pi_{old}) - \frac{2\gamma \epsilon_{model}}{(1-\gamma)^2} - \frac{2\gamma r_{max} \epsilon_{\pi}}{(1-\gamma)^2}$$
   其中 $\epsilon_{model}$ 是模型误差,$\epsilon_\pi$ 是策略偏移。结论:**rollout 越长,模型误差的惩罚项越大**。极端情况下,真实 off-policy 数据总是优于模型生成的 on-policy 数据。

2. **实用方案**:用**极短 rollout**(通常 1 步,最多几步)从真实数据 buffer 中分支出来:
   ```
   真实数据 buffer: (s_0, a_0, r_0, s_1, a_1, r_1, ...)
                        ↓ 从 s_i 出发做 1 步模型 rollout
                    模型数据: (\hat{s}_{i+1}, \hat{r}_{i+1})
   混合比例: 真实数据 95% + 模型数据 5%
   ```

3. **集成动力学模型**:沿用 [[02-模块/Model-Based/PETS|PETS]] 的 5 个网络集成,但只用来做极短预测。

### 关键公式/架构

$$\text{混合比例: } f = \frac{\text{model rollouts}}{\text{total updates}} \approx 0.05$$

**Schedule**:训练早期 rollout 长度 = 1(模型不够好),训练后期逐渐增加到 5(模型变好)。

### 优缺点

- **优点**:样本效率接近 [[02-模块/Model-Based/PETS|PETS]],但最终性能更好(因为短 rollout 避免了模型误差累积);计算成本比 [[02-模块/Model-Based/PETS|PETS]] 低(不需要在线 CEM 规划);理论上有策略改进保证;容易和现有 Model-Free 算法([[02-模块/Actor-Critic/SAC|SAC]])结合
- **缺点**:需要大量真实数据做 buffer,在极早期(数据很少时)效果一般;rollout 长度需要手动 schedule;本质上还是"用模型做数据增强",没有充分利用模型的规划能力
- **演化位置**:Dynasty 路线的成熟之作。核心贡献是理论回答了"该多信任模型"这个问题,给出了工程上简洁有效的方案。[[02-模块/Model-Based/MBPO|MBPO]] 成为后来很多 MBRL 工作的 baseline

---

## 9. Dreamer v1 — Hafner et al. (2020)

### 来源与动机

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi 发表在 ICLR 2020(arXiv:1912.01603)。论文标题: "Dream to Control: Learning Behaviors by Latent Imagination"。

核心问题:World Models(2018)开创了潜空间想象的范式,但有两个瓶颈:① RNN 预测能力有限;② Controller 用进化策略优化,效率低。Dreamer 要解决这两个问题。

### 核心创新

1. **RSSM (Recurrent State-Space Model)**:替代 World Models 的 VAE + MDN-RNN:
   - **确定性状态** $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$ (GRU 输出)
   - **随机性状态** $s_t \sim p(s_t | h_t)$ (先验) / $s_t \sim q(s_t | h_t, o_t)$ (后验,训练时用)
   - 推理时只用先验 $p$,不用观测 $o_t$ → 实现纯潜空间想象

2. **Actor-Critic 在潜空间**:在 RSSM 的潜空间中 rollout 想象轨迹,用 **Actor-Critic** (而非进化策略)优化策略:
   - Actor: $\pi(a_t | s_t)$,通过 reparameterization trick 反向传播梯度
   - Critic: $V(s_t)$,用 $\lambda$-return 估计价值

### 关键公式/架构

**RSSM 转移**:
$$h_t = f_\phi(h_{t-1}, s_{t-1}, a_{t-1})$$
$$\text{Prior: } p_\phi(s_t | h_t) = \mathcal{N}(\mu_p(h_t), \sigma_p(h_t))$$
$$\text{Posterior: } q_\phi(s_t | h_t, o_t) = \mathcal{N}(\mu_q(h_t, e_\psi(o_t)), \sigma_q(h_t, e_\psi(o_t)))$$

**训练目标** (ELBO):
$$\mathcal{L} = \sum_t \mathbb{E}_{q}[\log p(o_t | s_t, h_t) + \log p(r_t | s_t, h_t) - \beta \cdot KL(q(s_t|h_t, o_t) \| p(s_t|h_t))]$$

### 优缺点

- **优点**:比 World Models 更强大(RSSM 同时有确定性和随机性状态);Actor-Critic 比进化策略高效得多;在 20 个视觉控制任务上超越 Model-Free 方法的样本效率
- **缺点**:仍需要 observation reconstruction(解码器),可能学到与决策无关的细节;长 horizon 想象时 RSSM 预测仍会退化;超参数需要调优
- **演化位置**:World Models 路线的重要升级。RSSM 成为后续 Dreamer v2/v3 的核心组件

---

## 10. MuZero — Schrittwieser et al. (2020)

### 来源与动机

Julian Schrittwieser 等人(DeepMind)发表在 Nature 2020。核心问题:AlphaGo 依赖围棋的完美模拟器(perfect simulator),MuZero 要在**不知道规则**的情况下达到超人水平。

### 核心创新

**价值等价模型 (Value-Equivalent Model)**:不预测完整的下一状态,只预测对 MCTS 搜索有用的三个量:

1. **表示网络 (Representation)**: $h(o_1, ..., o_t) \to s^1$ (将观测编码为初始隐状态)
2. **动力学网络 (Dynamics)**: $g(s^k, a^k) \to (s^{k+1}, r^{k+1})$ (隐状态间的转移)
3. **预测网络 (Prediction)**: $f(s^k) \to (p^k, v^k)$ (策略先验和价值)

**关键洞察**:模型不需要重建观测,只需要在"对规划有价值的维度上"准确。这绕过了观测重建的难题,也避免了模型在无关维度上浪费容量。

### 关键公式/架构

$$s^1 = h(o_1, \ldots, o_t)$$
$$s^{k+1}, r^{k+1} = g(s^k, a^k)$$
$$p^k, v^k = f(s^k)$$

**训练损失**:
$$\mathcal{L} = \sum_k \left[ -\pi_k \log p^k + (z_k - v^k)^2 + (u_k - r^k)^2 \right]$$

其中 $\pi_k$ 是 MCTS 搜索策略,$z_k$ 是搜索得到的价值,$u_k$ 是真实奖励。

**规划**:用学到的 $(h, g, f)$ 在 MCTS 中展开搜索树,替代 AlphaGo 的真实模拟器。

### 优缺点

- **优点**:不需要环境规则,纯从数据学习;在围棋、国际象棋、将棋、Atari 上均达到超人水平;价值等价思想深刻——不要求模型"正确",只要求"有用"
- **缺点**:计算成本极高(训练需要大量 TPU);MCTS 搜索在推理时也很昂贵;隐状态不可解释,难以诊断问题;在需要精确物理推理的任务中可能不如显式动力学模型
- **演化位置**:MCTS 路线的巅峰之作。将 AlphaGo 的"搜索 + 神经网络"升级为"搜索 + 学到的模型 + 神经网络"。价值等价思想影响深远

---

## 11. Dreamer v2 — Hafner et al. (2021)

### 来源与动机

Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba 发表在 ICLR 2022(arXiv:2010.02193)。论文标题: "Mastering Atari with Discrete World Models"。

核心问题:Dreamer v1 的潜空间是连续的(高斯分布),但 Atari 等游戏环境有离散的本质结构(物体出现/消失)。能否用**离散**潜变量构建更好的世界模型?

### 核心创新

1. **离散世界模型**:用 one-hot 分类变量替代连续高斯:
   $$s_t \sim \text{Categorical}(K \text{ classes})$$
   每个潜变量从 $K$ 个类别中选一个(通常 $K=32$, 用 $N=32$ 个独立的分类变量)。

2. **Hindsight Replay**:用后验 $q(s_t|h_t, o_t)$ 做 replay 训练,而非先验 $p(s_t|h_t)$。这解决了离散变量无法用 reparameterization trick 的问题。

3. **对称 [[01-原子/KL散度|KL 散度]]**:训练时使用 symmetric KL 来稳定学习。

### 关键架构

**RSSM (离散版)**:
$$h_t = \text{GRU}(h_{t-1}, s_{t-1}, a_{t-1})$$
$$s_t \sim \text{Categorical}(\text{softmax}(W h_t))$$

**首个在 Atari 上从像素学习达到与 IQN(MF SOTA) 相当性能的 MBRL 方法**。

### 优缺点

- **优点**:离散潜变量更适合有离散结构的环境;在 Atari 上首次让 MBRL 达到 MF SOTA;训练更稳定
- **缺点**:离散表示可能丢失连续控制任务所需的精度;计算成本仍然较高
- **演化位置**:Dreamer 路线的重要进化,验证了离散潜变量的有效性

---

## 12. IRIS — Micheli, Alonso & Fleuret (2023)

### 来源与动机

Vincent Micheli, Eloi Alonso, François Fleuret 发表在 ICLR 2023(arXiv:2209.00588, Notable Top 5%)。论文标题: "Transformers are Sample-Efficient World Models"。

核心问题:RSSM 等循环模型的序列建模能力受限于固定大小的隐状态。Transformer 在语言建模中展示了卓越的长程建模能力——能否用 Transformer 做世界模型?

### 核心创新

1. **离散自编码器 (Discrete Autoencoder)**:用 VQ-VAE 将每帧观测编码为离散的 token 序列(每帧 4 个 token)。

2. **自回归 Transformer 动力学模型**:将观测 token 和动作 token 交织成一个序列,用 GPT 式 Transformer 自回归预测下一个 token:
   $$P(z_{t+1}^1, z_{t+1}^2, z_{t+1}^3, z_{t+1}^4 | z_{1:t}, a_{1:t})$$

3. **MuZero 式策略优化**:不用潜空间想象,而是用 Transformer 生成 rollout,然后用 [[02-模块/Policy-Based/PPO|PPO]] 风格的[[01-原子/策略梯度|策略梯度]]优化。

### 关键公式/架构

**Token 化**:
$$o_t \xrightarrow{\text{VQ-VAE}} [z_t^1, z_t^2, z_t^3, z_t^4] \in \{1, \ldots, V\}^4$$

**序列**:
$$[z_1^1, z_1^2, z_1^3, z_1^4, a_1, z_2^1, z_2^2, z_2^3, z_2^4, a_2, \ldots]$$

**Atari 100k 基准**:仅 2 小时游戏数据,IRIS 在 26 个 Atari 游戏上达到 1.046 的人类归一化得分,超过人类 10 个游戏。是**不使用 lookahead search 的方法中的 SOTA**。

### 优缺点

- **优点**:Transformer 的长程依赖建模能力远超 RNN;离散 token 避免了连续潜空间的分布偏移问题;样本效率极高
- **缺点**:Transformer 推理成本随序列长度 $O(L^2)$ 增长;离散 token 化限制了泛化能力;只在 Atari 上验证,连续控制任务效果未知;VQ-VAE 训练不稳定
- **演化位置**:首次将 Transformer 引入世界模型,开辟了"Transformer as World Model"新方向。直接影响了后来的 TWM、TD-MPC2 的 Transformer 变体等

---

## 13. Dreamer v3 — Hafner et al. (2023)

### 来源与动机

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap 发表(arXiv:2301.04104)。论文标题: "Mastering Diverse Domains through World Models"。

核心问题:Dreamer v1/v2 在不同任务领域需要不同的超参数,限制了实用性。能否设计一个**通用的、免调参**的 MBRL 算法?

### 核心创新

Dreamer v3 是 Dreamer 系列的集大成之作,核心贡献是**一系列鲁棒性技术**使其在 150+ 个不同领域的任务上,用**同一组超参数**就能达到优秀性能。

1. **Symlog 值预测**:对价值函数取 symlog 变换,避免不同领域奖励尺度差异的影响:
   $$\text{symlog}(x) = \text{sign}(x) \cdot \log(1 + |x|)$$

2. **自由比特 KL (Free Bits KL)**:对 KL 散度的每个维度独立设置阈值,避免后验坍缩:
   $$KL_{free}(q \| p) = \sum_i \max(KL_i(q \| p), \tau)$$

3. **归一化与平衡**:对潜空间各维度做归一化,平衡重建损失和 KL 损失的权重。

4. **数据增强**:对称变换(symmetry transformations)增加数据多样性。

**里程碑成就**:**首个从零开始在 Minecraft 中收集钻石的算法**(不需要人类数据或课程学习)。

### 关键架构

沿用 Dreamer v1/v2 的 RSSM + Actor-Critic 框架,核心改进在鲁棒性:

$$\mathcal{L}_{world} = -\mathbb{E}_q[\log p(o|s,h)] - \mathbb{E}_q[\log p(r|s,h)] + \beta \cdot KL_{free}(q \| p)$$
$$\mathcal{L}_{actor} = -\mathbb{E}[\hat{V}^\lambda(s)] - \eta \cdot H[\pi(\cdot|s)]$$
$$\mathcal{L}_{critic} = \mathbb{E}[(\text{symlog}(\hat{V}^\lambda) - \text{symlog}(V(s)))^2]$$

### 优缺点

- **优点**:真正做到了跨领域通用性(单一超参数配置);在 Minecraft 中实现了里程碑式成就;鲁棒性技术系统且有理论支撑
- **缺点**:计算成本仍然很高(需要 GPU 训练世界模型 + 想象);RSSM 的长期预测仍有退化;没有在竞争性多智能体场景中验证
- **演化位置**:Dreamer 路线的成熟之作,标志着 MBRL 从"需要大量调参的研究工具"走向"开箱即用的通用算法"

---

## 14. TD-MPC2 — Hansen et al. (2024)

### 来源与动机

Nicklas Hansen, Hao Su, Xiaolong Wang 发表在 ICLR 2024(arXiv:2310.16828)。论文标题: "TD-MPC2: Scalable, Robust World Models for Continuous Control"。

核心问题:现有 MBRL 方法要么局限于单一任务域,要么需要针对不同域调参。TD-MPC 的初版展示了"隐式世界模型 + 局部轨迹优化"的有效性,但能否做到**跨域、可扩展、单一超参数**?

### 核心创新

1. **隐式世界模型 (Implicit World Model)**:不预测观测,只预测**潜状态的 TD 目标**:
   $$z_{t+1} = f_\theta(z_t, a_t)$$
   训练目标:让 $z_t$ 编码对预测未来奖励有用的信息(TD learning),而非重建观测。

2. **多任务扩展**:单一 317M 参数模型跨 80 个任务、多个任务域、多种 embodiment、多种动作空间工作。

3. **关键改进**:
   - 集成表示学习(ensemble representations)
   - 基于模型的 actor-critic
   - 跨任务共享的通用架构

### 关键公式

$$\mathcal{L}_{model} = \sum_{t} \left\| z_t - \text{sg}(z_t^{target}) \right\|^2 + \left\| r_t - \hat{r}_t \right\|^2 + \left\| d_t - \hat{d}_t \right\|^2$$

其中 $z_t^{target}$ 是 TD 目标,$\text{sg}$ 是 stop-gradient。

**规划**:在潜空间中用 MPPI(Model Predictive Path Integral)做局部轨迹优化。

### 优缺点

- **优点**:跨域泛化能力强;单一超参数;模型随规模增加能力提升(类 scaling law);推理速度快(不需要解码观测)
- **缺点**:隐式模型不可解释;在需要精确视觉推理的任务中可能不足;大规模训练需要大量数据
- **演化位置**:结合了 [[02-模块/Model-Based/PETS|PETS]]/[[02-模块/Model-Based/MBPO|MBPO]] 的"实用主义"(短 rollout + 不确定性)和 MuZero 的"价值等价"(不重建观测),是两条路线的融合之作

---

## 15. Genie — Bruce et al. / DeepMind (2024)

### 来源与动机

Jake Bruce 等人(Google DeepMind)发表(arXiv:2402.15391, 2024 年 2 月)。论文标题: "Genie: Generative Interactive Environments"。

核心问题:现有世界模型要么需要动作标签训练(如 Dreamer),要么只能在固定环境中工作。能否从**无标签互联网视频**中学出一个通用的、可交互的世界模型?

### 核心创新

**首个从无标签视频学习的生成式交互环境 (11B 参数基础世界模型)**:

1. **时空视频 tokenizer (Spatiotemporal Video Tokenizer)**:将视频编码为离散 token,同时捕捉空间和时间信息。

2. **潜在动作模型 (Latent Action Model)**:核心创新——从视频中自动发现"动作"。通过对比相邻帧,学出一个潜在动作空间,无需任何真实动作标签:
   $$a_t^{latent} = \text{LAM}(o_t, o_{t+1})$$

3. **自回归动力学模型 (Autoregressive Dynamics Model)**:给定当前帧 token 和潜在动作,预测下一帧 token。

### 关键能力

- 用文本、图片、照片、甚至手绘草图生成无限多样的可交互虚拟世界
- 用户逐帧控制环境(无需真实动作标签)
- 学到的潜在动作空间可用于训练**从未见过视频中**模仿行为的 agent

### 优缺点

- **优点**:无需标签,从互联网视频扩展;11B 参数展示了基础世界模型的规模效应;潜在动作模型解决了无标签问题
- **缺点**:推理速度慢(自回归生成);生成质量仍有瑕疵(长序列退化);潜在动作空间不一定对应真实的可控维度;计算资源需求巨大
- **演化位置**:标志着世界模型从"特定环境的模拟器"走向"基础世界模型"(foundation world model)。是 World Models(2018) 理念的极致扩展

---

## 16. GameNGen — Valevski et al. / Google (2024)

### 来源与动机

Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter(Google)发表在 ICLR 2025(arXiv:2408.14837, 2024 年 8 月)。论文标题: "Diffusion Models Are Real-Time Game Engines"。

核心问题:传统游戏引擎依赖显式物理模拟和渲染管线。能否用一个**神经网络完全替代游戏引擎**,实时生成可交互的游戏画面?

### 核心创新

**首个完全由神经网络驱动的游戏引擎**:

1. **两阶段训练**:
   - **阶段 1**:训练一个 RL agent 玩 DOOM,记录游戏过程(观测 + 动作)
   - **阶段 2**:训练一个扩散模型(diffusion model),给定历史帧序列和动作,预测下一帧

2. **实时交互**:在单个 TPU 上达到 20 FPS,可以持续多分钟交互。

3. **条件增强 (Conditioning Augmentation)**:解决自回归生成的长序列退化问题,对条件输入做随机扰动以增强稳定性。

4. **解码器微调**:改善视觉细节(文字、UI 元素)的保真度。

**核心指标**:PSNR 29.4(接近有损 JPEG 压缩质量);人类评估者在 5 分钟自回归生成后,仅略好于随机猜测地区分真实游戏和模拟。

### 优缺点

- **优点**:实时可交互(20 FPS);质量极高(接近真实);展示了扩散模型作为世界模型的潜力
- **缺点**:只在 DOOM 上验证;需要 RL agent 先玩游戏收集数据;泛化到新游戏需要重新训练;物理一致性不完美(长时间后可能出现不合理现象)
- **演化位置**:首次将扩散模型用作游戏引擎/世界模型,开创了"视频生成 = 世界模拟"的新范式。直接影响了 Genie 2、Sora 类世界模型的研究方向

---

## 17. 视频生成作为世界模型:Genie 2 / UniSim / Sora 类 (2024-2026)

### 背景与趋势

2024-2026 年,视频生成模型的爆发式进步催生了"视频生成模型 = 世界模型"的研究范式。核心假设:如果一个模型能生成逼真的、物理一致的视频,那它实质上已经学到了世界的动力学。

### Genie 2 (DeepMind, 2024 年底)

Genie 的升级版,能从单张图片生成一致、持久的 3D 可交互世界。关键改进:
- 更好的长时间一致性(consistent world state)
- 更丰富的交互能力
- 支持多种环境类型

### UniSim (Waabi, 2023)

Ze Yang 等人(Waabi, CVPR 2023 Highlight, arXiv:2308.01898)。论文标题: "UniSim: A Neural Closed-Loop Sensor Simulator"。

虽然名字相同,但这里的 UniSim 是自动驾驶领域的**闭环传感器模拟器**:
- 从单次驾驶日志构建场景的神经特征网格
- 支持闭环评估:自动驾驶系统的决策会影响模拟结果
- 可模拟 LiDAR 和相机数据

### Sora 类模型作为世界模型

OpenAI 的 Sora(2024 年初发布)展示了大规模视频生成模型的能力。虽然 Sora 本身不是为 RL 设计的,但研究社区迅速认识到其作为世界模型的潜力:

1. **Mei et al. (2026)** 的综述 "Video Generation Models in Robotics"(arXiv:2601.07823)系统梳理了视频模型在机器人中的应用:
   - **数据生成**:用视频模型生成训练数据
   - **动力学建模**:视频模型预测未来帧作为动力学模型
   - **奖励建模**:从视频预测中提取奖励信号
   - **视觉规划**:在视频空间中规划动作序列
   - **策略评估**:用视频模型评估策略的好坏

2. **挑战**:
   - **物理不一致性**:生成视频可能违反物理定律
   - **指令跟随不准确**:条件控制不精确
   - **幻觉**:生成不存在的物体或现象
   - **计算成本**:训练和推理都需要大量资源

### LoopWM — Lu et al. (2026)

Hongyuan Adam Lu 等人(arXiv:2606.18208, 2026 年 6 月)。论文标题: "Looped World Models"。

**最新进展**:首个将**循环架构**引入世界模型的工作。核心思想:
- 用参数共享的 Transformer block 迭代精炼潜状态
- 比传统方法参数效率高 100 倍
- 自适应计算深度——简单预测用少量迭代,复杂预测用多量迭代
- 提出"迭代潜深度"(iterative latent depth)作为世界模拟的新 scaling 维度

### SMWM — Ivashkov et al. (2026)

Petr Ivashkov, Randall Balestriero, Bernhard Scholkopf(arXiv:2606.20104, 2026 年 6 月)。论文标题: "Sensorimotor World Models: Perception for Action via Inverse Dynamics"。

**最新进展**:
- JEPA 风格的潜空间世界模型,加入逆动力学正则化
- 防止表示坍缩,同时让潜空间对齐动作
- 不需要冻结编码器、EMA 或复杂正则化
- 从离线无奖励轨迹中学习

### BRICKS-WM — Zhang et al. (2026)

Shaowei Zhang 等人(arXiv:2606.16489, 2026 年 6 月)。论文标题: "BRICKS-WM: Building Reusability via Interface Composition Kinetics for Structured World Models"。

**最新进展**:
- 将世界模型模块化:Agent 模块 + Background 模块 + 潜接口
- 背景动力学不依赖 agent 动力学 → 可复用
- 换 agent 不需要重训整个世界模型

---

## 18. 2025-2026 最新进展

### GIRL — Hiremath (2026)

Prakul Sunil Hiremath(arXiv:2604.07426, 2026 年 4 月)。论文标题: "GIRL: Generative Imagination Reinforcement Learning via Information-Theoretic Hallucination Control"。

**核心贡献**:
- 用 DINOv2 基础模型的跨模态信号锚定潜转移动力学
- 不确定性自适应信任区域:将 KL 正则化解释为约束优化的拉格朗日乘子
- 理论推导了值差距界(value gap bound),使用 Performance Difference Lemma + IPM
- 在 DMControl、Adroit、Meta-World 上比 Dreamer v3 减少 38-61% 的 rollout 漂移
- 在稀疏奖励任务上超越 TD-MPC2

### FlowMPC — Hamel (2026)

Chandon Hamel(arXiv:2606.16286, 2026 年 6 月)。论文标题: "FlowMPC: Improving Flow Matching policies with World Models"。

- 将行为克隆的 Flow Matching 策略与学到的世界模型结合
- 推理时用 MPPI 在模型中规划,改进行为克隆策略的动作
- 在 ManiSkill 操作任务上提升了 Flow Matching 策略的成功率

### TransZero — Malmsten & Bohmer (2025)

Emil Malmsten, Wendelin Bohmer(arXiv:2509.11233, 2025)。论文标题: "TransZero: Parallel Tree Expansion in MuZero using Transformer Networks"。

- 用 Transformer 替代 MuZero 的循环动力学网络
- 同时生成多个潜未来状态(并行展开)
- 比 MuZero 快 11 倍(wall-clock time),同时保持样本效率

---

## 19. 三大主线对比总结

### 主线一:Dyna → [[02-模块/Model-Based/PETS|PETS]] → [[02-模块/Model-Based/MBPO|MBPO]](学动力学 + 短 rollout)

| 方法 | 年份 | 模型类型 | Rollout 长度 | 不确定性处理 | 规划方式 |
|------|------|----------|-------------|-------------|---------|
| [[02-模块/Model-Based/Dyna-Q|Dyna-Q]] | 1991 | Tabular | 1 步 | 无 | Q-learning |
| [[02-模块/Model-Based/PETS|PETS]] | 2018 | 集成 NN | 多步(H=25-40) | 集成方差 | CEM |
| [[02-模块/Model-Based/MBPO|MBPO]] | 2019 | 集成 NN | 1-5 步 | 集成方差 + 短截断 | [[02-模块/Actor-Critic/SAC|SAC]] 策略 |

**核心逻辑**:模型不可信太远 → 从真实数据出发做短 rollout → 模型误差被截断。

### 主线二:World Models → Dreamer → IRIS(潜空间想象)

| 方法 | 年份 | 潜空间 | 序列模型 | 策略优化 |
|------|------|--------|---------|---------|
| World Models | 2018 | VAE 连续 | MDN-RNN | CMA-ES(进化) |
| Dreamer v1 | 2020 | RSSM 连续 | GRU | Actor-Critic |
| Dreamer v2 | 2021 | RSSM 离散 | GRU | Actor-Critic |
| IRIS | 2023 | VQ-VAE 离散 | Transformer | [[02-模块/Policy-Based/PPO|PPO]] 风格 |
| Dreamer v3 | 2023 | RSSM 离散 | GRU | Actor-Critic(免调参) |

**核心逻辑**:在低维潜空间中想象 → 误差在高维像素空间中被压缩 → 想象成本低且可微分。

### 主线三:MCTS → AlphaGo → MuZero(搜索 + 学习)

| 方法 | 年份 | 模型来源 | 搜索方式 | 评估方式 |
|------|------|---------|---------|---------|
| MCTS/UCT | 2006 | 环境模拟器 | UCB 树搜索 | 随机 rollout |
| AlphaGo | 2016 | 环境模拟器 + 数据 | MCTS + NN 指导 | 策略网络 + 价值网络 |
| MuZero | 2020 | 学到的模型 | MCTS 在潜空间 | 学到的价值网络 |
| TD-MPC2 | 2024 | 隐式模型 | MPPI 局部搜索 | TD 学习 |

**核心逻辑**:不要求模型完全准确 → 搜索过程自动纠正模型小错误 → 只关注"对决策有价值"的维度。

---

## 20. 模型误差对抗策略全景

| 策略 | 原理 | 代表方法 | 适用场景 |
|------|------|----------|---------|
| **短 rollout** | 从真实数据分支,只预测 1-5 步,误差来不及累积 | [[02-模块/Model-Based/Dyna-Q|Dyna-Q]], [[02-模块/Model-Based/MBPO|MBPO]] | 需要大量真实数据 |
| **集成不确定性** | 多个模型的分歧量化不确定性,分歧大时保守 | [[02-模块/Model-Based/PETS|PETS]], [[02-模块/Model-Based/MBPO|MBPO]] | 中等规模问题 |
| **潜空间压缩** | 在低维潜空间中想象,高维误差被压缩 | World Models, Dreamer | 高维观测(图像) |
| **价值等价** | 不预测完整状态,只预测对决策有用的量 | MuZero, TD-MPC2 | 有搜索/规划需求 |
| **离散 token 化** | 离散表示避免连续漂移,更稳定 | Dreamer v2, IRIS | 有离散结构的环境 |
| **扩散模型** | 逐步去噪生成,天然处理多模态和不确定性 | GameNGen, Genie | 高保真视觉生成 |
| **循环精炼** | 参数共享 + 自适应深度,用迭代替代深度 | LoopWM (2026) | 需要不同复杂度的预测 |
| **模块化** | 将世界分解为独立模块,各自学习动力学 | BRICKS-WM (2026) | 需要复用/组合的场景 |
| **基础模型锚定** | 用预训练基础模型的表征约束潜空间 | GIRL (2026) | 防止潜空间漂移 |

---

## 21. 参考文献

### 经典基础
1. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
2. Sutton, R. S. (1991). "Dyna, an Integrated Architecture for Learning, Planning, and Reacting." *ACM SIGART Bulletin*, 2(4), 160-163.
3. Kocsis, L., & Szepesvári, C. (2006). "Bandit Based Monte-Carlo Planning." *ECML*.

### [[02-模块/Model-Based/PETS|PETS]]-[[02-模块/Model-Based/MBPO|MBPO]] 系
4. Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." *NeurIPS*. arXiv:1805.12114.
5. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). "When to Trust Your Model: Model-Based Policy Optimization." *NeurIPS*. arXiv:1906.08253.

### World Models-Dreamer 系
6. Ha, D., & Schmidhuber, J. (2018). "World Models." arXiv:1803.10122.
7. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). "Dream to Control: Learning Behaviors by Latent Imagination." *ICLR*. arXiv:1912.01603.
8. Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). "Mastering Atari with Discrete World Models." *ICLR*. arXiv:2010.02193.
9. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). "Mastering Diverse Domains through World Models." arXiv:2301.04104.

### Transformer 世界模型
10. Micheli, V., Alonso, E., & Fleuret, F. (2023). "Transformers are Sample-Efficient World Models." *ICLR* (Notable Top 5%). arXiv:2209.00588.

### MCTS-MuZero 系
11. Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." *Nature*, 588, 604-609.
12. Schrittwieser, J., et al. (2021). "Online and Offline Reinforcement Learning by Planning with a Learned Model." arXiv:2104.06294.

### TD-MPC 系
13. Hansen, N., Su, H., & Wang, X. (2024). "TD-MPC2: Scalable, Robust World Models for Continuous Control." *ICLR*. arXiv:2310.16828.

### 视频生成作为世界模型
14. Bruce, J., et al. (2024). "Genie: Generative Interactive Environments." arXiv:2402.15391.
15. Valevski, D., Leviathan, Y., Arar, M., & Fruchter, S. (2024). "Diffusion Models Are Real-Time Game Engines." *ICLR 2025*. arXiv:2408.14837.
16. Yang, Z., et al. (2023). "UniSim: A Neural Closed-Loop Sensor Simulator." *CVPR Highlight*. arXiv:2308.01898.
17. Mei, Z., et al. (2026). "Video Generation Models in Robotics -- Applications, Research Challenges, Future Directions." arXiv:2601.07823.

### 2025-2026 最新工作
18. Lu, H. A., et al. (2026). "Looped World Models." arXiv:2606.18208.
19. Ivashkov, P., et al. (2026). "Sensorimotor World Models: Perception for Action via Inverse Dynamics." arXiv:2606.20104.
20. Zhang, S., et al. (2026). "BRICKS-WM: Building Reusability via Interface Composition Kinetics for Structured World Models." arXiv:2606.16489.
21. Hiremath, P. S. (2026). "GIRL: Generative Imagination Reinforcement Learning via Information-Theoretic Hallucination Control." arXiv:2604.07426.
22. Hamel, C. (2026). "FlowMPC: Improving Flow Matching policies with World Models." arXiv:2606.16286.
23. Malmsten, E., & Böhmer, W. (2025). "TransZero: Parallel Tree Expansion in MuZero using Transformer Networks." arXiv:2509.11233.

---

> **文档版本**:v1.0 | **最后更新**:2026-06-21 | **作者**:研究综述
