---
tags:
  - #model-free
  - #actor-critic
  - #off-policy
  - #online-rl
  - #continuous
  - #function-approximation
created: 2026-06-25
---

## [Link] 知识图谱链接

### 相关算法
- [[TD3]]
- [[SAC]]
- [[DQN]]

### 分类维度
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|Off-Policy]]
- [[00-索引/按动作空间|Continuous-Actions]]

---

# Deep Deterministic Policy Gradient (DDPG)

> 深度确定性策略梯度：将 DQN 的成功经验迁移到连续动作空间的开创性算法。Actor 输出确定性动作，Critic 用 DQN 风格更新。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：Actor 通过确定性策略梯度优化
- **确定性策略** $\mu_\theta(s)$：Actor 直接输出动作（不是概率分布）
- [[01-原子/Q-Learning]]：Critic 用 DQN 风格的 TD 更新
- [[01-原子/经验回放]]：Replay Buffer 存储历史转移，提升样本效率
- [[01-原子/目标网络]]：Soft update 稳定训练
- [[01-原子/重要性采样]]：Off-policy 数据复用的理论基础

---

## 核心创新

DDPG 解决的核心问题：**DQN 在离散动作空间很成功，但连续动作空间怎么办？**

DQN 的核心操作是 $\arg\max_a Q(s,a)$——在所有可能的动作中找 Q 值最大的那个。在离散动作空间（比如"左/右/不动"）中，这很简单：遍历所有动作就行。但在连续动作空间（比如机械臂的关节角度，取值范围 $[-\pi, \pi]$），你不可能遍历无限多个动作来找最大值。

**核心思想**（Lillicrap et al., 2016）：既然找不到最大值，就让一个 Actor 网络直接**学**哪个动作最好。Actor 网络 $\mu_\theta(s)$ 直接输出一个确定性动作，Critic 网络 $Q_\phi(s,a)$ 负责评价这个动作好不好。

**确定性策略梯度定理**（Silver et al., 2014）证明了：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\beta} [\nabla_a Q(s,a)|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)]$$

这个公式的直觉：**Critic 告诉 Actor"你应该往哪个方向调整动作"，Actor 就朝那个方向走。**

---

## 算法流程

### 1. 初始化

```
1. Actor 网络 μ_θ（确定性策略）
2. Critic 网络 Q_φ
3. 目标网络：μ_{θ'}, Q_{φ'}（参数从主网络拷贝）
4. Replay Buffer D
5. 噪声过程 N（如 Ornstein-Uhlenbeck 过程）用于探索
```

### 2. 数据收集

```
对于每个 episode:
    1. 初始化噪声过程 N
    2. 观察初始状态 s_0
    3. 对于每个时间步 t:
        a. 选择动作：a_t = μ_θ(s_t) + N_t    // 确定性动作 + 探索噪声
        b. 执行 a_t，观察 r_t, s_{t+1}
        c. 存储 (s_t, a_t, r_t, s_{t+1}) 到 Replay Buffer D
```

### 3. 网络更新（从 Buffer 中采样 Mini-batch）

```
从 D 中随机采样 N 个转移 {(s_i, a_i, r_i, s'_i)}：

1. 更新 Critic：
   y_i = r_i + γ · Q_{φ'}(s'_i, μ_{θ'}(s'_i))
   L_Q = (1/N) Σ (Q_φ(s_i, a_i) - y_i)^2

2. 更新 Actor：
   L_μ = -(1/N) Σ Q_φ(s_i, μ_θ(s_i))
   （最小化负的 Q 值 = 最大化 Q 值）

3. Soft Update 目标网络：
   θ' ← τ · θ + (1 - τ) · θ'
   φ' ← τ · φ + (1 - τ) · φ'
   （τ 通常取 0.005）
```

### 4. 重复

---

## 关键公式

### 确定性策略梯度（Actor 更新）

$$\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_a Q_\phi(s, a)|_{s=s_i, a=\mu_\theta(s_i)} \cdot \nabla_\theta \mu_\theta(s)|_{s=s_i}$$

逐项拆解：

1. **$\nabla_a Q_\phi(s, a)$**：Critic 对动作的梯度——"动作往哪个方向调，Q 值会变大？"
2. **$\nabla_\theta \mu_\theta(s)$**：Actor 对参数的梯度——"参数怎么调，才能改变输出的动作？"
3. **链式法则**：两者相乘 = "参数怎么调，才能让 Critic 给更高的分？"

**直觉**：Critic 是"老师"，告诉 Actor "你的动作应该往左偏一点"；Actor 是"学生"，调整自己让下次输出更靠左。

### Critic Loss（DQN 风格 TD 更新）

$$L(\phi) = \frac{1}{N} \sum_i (Q_\phi(s_i, a_i) - y_i)^2$$

$$y_i = r_i + \gamma \cdot Q_{\phi'}(s'_i, \mu_{\theta'}(s'_i))$$

- $y_i$ 是**目标 Q 值**，用目标网络计算（避免自举不稳定）
- 与 DQN 的区别：DQN 用 $\max_a Q(s',a)$，DDPG 用 $Q(s', \mu(s'))$——因为连续空间不能做 $\max$

### Soft Update（目标网络更新）

$$\theta' \leftarrow \tau \theta + (1-\tau)\theta'$$
$$\phi' \leftarrow \tau \phi + (1-\tau)\phi'$$

- $\tau$ 很小（通常 0.005），意味着目标网络变化很慢
- **为什么不用 Hard Update？** Hard Update（如 DQN 每隔几千步拷贝一次参数）会导致目标值突变，训练震荡。Soft Update 让目标值平滑变化。

---

## 探索机制

DDPG 是**确定性策略**——对于同一个状态，永远输出同一个动作。这意味着必须**额外添加噪声**来探索：

$$a_t = \mu_\theta(s_t) + \mathcal{N}_t$$

### 常用噪声类型

1. **高斯噪声**：$\mathcal{N}_t \sim \mathcal{N}(0, \sigma^2 I)$，简单直接
2. **Ornstein-Uhlenbeck (OU) 噪声**：
   $$d\mathcal{N}_t = \theta(\mu - \mathcal{N}_t)dt + \sigma dW_t$$
   - 有时间相关性（动量效应），适合物理控制任务
   - 比如推车：如果上一步往左推了，下一步可能继续往左推，而不是突然换方向
3. **Parameter Space Noise**：在参数空间而非动作空间加噪声

---

## 优缺点

### 优点
- **连续动作空间**：首次将深度 RL 成功应用到连续控制
- **Off-policy**：使用 Replay Buffer，样本效率高
- **架构清晰**：Actor 输出动作、Critic 评价——分工明确
- **训练稳定**：目标网络 + Soft Update 保证训练不发散

### 缺点
- **Q 值高估**：Critic 倾向于高估 Q 值（[[01-原子/Q-Learning|Q-Learning]] 的固有问题），导致策略学偏
- **超参数敏感**：学习率、噪声参数、$\tau$ 等对性能影响大
- **探索不足**：确定性策略 + 加噪声的探索方式比较粗糙
- **脆弱性**：在某些环境中容易崩溃，难以复现
- **被 TD3 和 SAC 超越**：TD3 解决了高估问题，SAC 引入了更好的探索机制

---

## 演化位置

**在连续控制演化链中的位置**：

```
DPG (2014)：确定性策略梯度定理（理论）
    ↓
DDPG (2016)：深度版本 + 经验回放 + 目标网络 ← 你在这里
    ↓
TD3 (2018)：双 Critic + 延迟更新 + 目标策略平滑
    ↓
SAC (2018)：最大熵 + 自适应温度
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象你在学开车。

- **DQN（离散）**：方向盘只有"左转/右转/不动"三档——太粗糙了，没法精确控制。
- **DDPG（连续）**：方向盘可以转到任意角度。Actor 是你的"肌肉记忆"——给定路况，自动输出一个方向盘角度。Critic 是你的"教练"——评价这个角度好不好。教练说"再往右打 5 度"，你的肌肉记忆就调整 5 度。
- **问题**：教练（Critic）可能过于乐观（高估 Q 值），导致你觉得自己的技术比实际好。TD3 请了两个教练，取保守的那个评价。SAC 则鼓励你多试不同开法（最大熵探索）。

**经典实验**：MuJoCo HalfCheetah-v2
- DDPG 通常能学到 ~3000 分
- TD3 改进后能到 ~9000 分
- SAC 能到 ~12000 分
- 说明 DDPG 的高估问题确实严重

---

## 代码片段

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 输出限制在 [-1, 1]
        )
        self.max_action = max_action
    
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, 
                 gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.gamma = gamma
        self.tau = tau
    
    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # 1. 更新 Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        # critic_optimizer.zero_grad(); critic_loss.backward(); critic_optimizer.step()
        
        # 2. 更新 Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # actor_optimizer.zero_grad(); actor_loss.backward(); actor_optimizer.step()
        
        # 3. Soft Update 目标网络
        for param, target_param in zip(self.actor.parameters(), 
                                        self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), 
                                        self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## 分类与相关算法

- **RL分类**：Model-free, Actor-Critic, Off-Policy
- **数据来源**：Off-Policy, Online-RL
- **动作空间**：Continuous（核心设计目标）
- **策略类型**：确定性策略（Deterministic Policy）
- **相关算法**：
  - [[02-模块/Value-Based/DQN|DQN]]（DDPG 的离散动作前身）
  - [[02-模块/Actor-Critic/TD3|TD3]]（修复 DDPG 的高估问题）
  - [[02-模块/Actor-Critic/SAC|SAC]]（最大熵版本，探索更好）

---

## 参考资源

- 论文：Lillicrap et al. (2016). "Continuous control with deep reinforcement learning." *ICLR*.
- 论文：Silver et al. (2014). "Deterministic Policy Gradient Algorithms." *ICML*.
- 代码：[OpenAI Spinning Up DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- 代码：[Stable Baselines3 DDPG](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)
- 代码：[CleanRL DDPG](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py)
