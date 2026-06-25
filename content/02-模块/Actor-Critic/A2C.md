---
tags:
  - #model-free
  - #actor-critic
  - #on-policy
  - #online-rl
  - #continuous
  - #function-approximation
created: 2026-06-25
---

## [Link] 知识图谱链接

### 相关算法
- [[A3C]]
- [[PPO]]

### 分类维度
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|On-Policy]]

---

# Advantage Actor-Critic (A2C)

> 优势演员-评论家：用同步并行环境收集数据，用优势函数降低策略梯度方差。A3C 的同步简化版，实际效果相当甚至更优。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：Actor 直接优化策略参数
- [[01-原子/优势函数]]：用 $A(s,a) = Q(s,a) - V(s)$ 代替纯回报，降低方差
- **Actor-Critic 架构**：Actor 学策略，Critic 学值函数 $V(s)$
- **同步并行采样**：多个环境副本同时运行，汇总梯度后统一更新
- [[01-原子/GAE]]（可选）：广义优势估计，平衡偏差与方差

---

## 核心创新

A2C 解决的核心问题：**A3C 的异步更新真的有必要吗？**

A3C (2016) 提出用多个异步 worker 并行探索环境，每个 worker 独立计算梯度并异步更新全局网络。这在当时被认为是"加速训练的关键"。但 OpenAI 在 2017 年发现：**把异步改成同步，效果不仅不差，反而更好。**

**核心思想**：与其让多个 worker 各自为战、用"过时的"全局参数计算梯度，不如等所有 worker 都跑完一步（或几步），**用同一份最新参数汇总梯度后统一更新**。

这个改动带来三个好处：
1. **训练更稳定**：所有梯度都基于同一套参数，没有"过期参数"的干扰
2. **GPU 利用率更高**：一个大批次同步更新比多个小批次异步更新更适配 GPU 的并行计算
3. **实现更简单**：不需要处理异步锁、参数冲突等问题

---

## 算法流程

### 1. 初始化

```
1. 创建全局网络 (Actor π_θ, Critic V_φ)
2. 创建 N 个环境副本 (env_1, ..., env_N)
```

### 2. 并行数据收集

```
在每个时间步 t:
    对于每个环境 i = 1, ..., N（并行执行）：
        a. 观察状态 s_t^i
        b. 从 π_θ(·|s_t^i) 采样动作 a_t^i
        c. 执行 a_t^i，得到 r_t^i, s_{t+1}^i, done_t^i
```

### 3. 计算优势

当收集了 T 步数据（或某个 episode 结束）时：

$$\hat{A}_t^i = R_t^i - V_\phi(s_t^i)$$

其中 $R_t^i$ 可以是：
- **n-step 回报**：$R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})$
- **GAE**：$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$

### 4. 汇总梯度并更新

$$\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot \hat{A}_t^i$$

$$L_\phi = \frac{1}{N \cdot T} \sum_{i=1}^{N} \sum_{t=0}^{T} (V_\phi(s_t^i) - R_t^i)^2$$

通常还会加一个**熵正则项**鼓励探索：

$$L_{\text{total}} = -\nabla_\theta J(\theta) + c_1 \cdot L_\phi - c_2 \cdot H(\pi_\theta(\cdot|s_t))$$

### 5. 重复

丢弃旧数据，用更新后的网络继续。

---

## 关键公式

### Actor Loss（策略损失）

$$L_{\text{actor}} = -\mathbb{E}_{t,i} [\log \pi_\theta(a_t^i | s_t^i) \cdot \hat{A}_t^i]$$

- **$\hat{A}_t^i > 0$**（这个动作比平均水平好）→ 增大 $\log \pi$ → 增大选这个动作的概率
- **$\hat{A}_t^i < 0$**（这个动作比平均水平差）→ 减小 $\log \pi$ → 减小选这个动作的概率

### Critic Loss（值函数损失）

$$L_{\text{critic}} = \mathbb{E}_{t,i} [(V_\phi(s_t^i) - R_t^i)^2]$$

标准均方误差回归，让 Critic 学会准确估计状态价值。

### 熵奖励

$$H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

- 熵越大 → 策略越随机 → 探索越充分
- 系数 $c_2$ 通常取 0.01

### 优势函数（核心降方差机制）

$$A(s, a) = Q(s, a) - V(s)$$

**直觉理解**：$V(s)$ 是"在这个状态下，平均能拿多少分"。$Q(s,a)$ 是"在这个状态下选这个动作，能拿多少分"。优势 $A = Q - V$ 就是"选这个动作比平均好多少"。如果 $A > 0$，说明这个动作超出平均，应该多试。

---

## A2C vs A3C：为什么同步更好？

| 特性 | A3C（异步） | A2C（同步） |
|------|------------|------------|
| 参数一致性 | Worker 用过时参数计算梯度 | 所有梯度基于最新参数 |
| GPU 利用 | 小批次异步更新，GPU 利用率低 | 大批次同步更新，GPU 利用率高 |
| 实现复杂度 | 需要异步框架、锁机制 | 简单直接 |
| 训练稳定性 | 梯度可能冲突（stale gradients）| 梯度一致，更稳定 |
| 实际效果 | 略差或相当 | 相当或更好 |

OpenAI 的实验表明：在 Atari 游戏上，A2C 用 4 个 GPU 的吞吐量是 A3C 用 8 个 CPU 的 3 倍以上。

---

## 优缺点

### 优点
- **低方差**：优势函数 + Critic 提供即时反馈，不用等 episode 结束
- **并行效率高**：同步大批次更新完美适配 GPU
- **实现简单**：比 A3C 简单得多，比 PPO 也简单
- **通用性好**：离散/连续动作空间都能用
- **样本效率**：On-policy 中算比较高的（多环境并行 = 数据量大）

### 缺点
- **样本效率不如 Off-policy**：数据用完即弃，不能复用
- **Critic 偏差**：如果 Critic 估计不准，优势函数会带偏差
- **探索有限**：主要靠策略自身的随机性和熵奖励，探索策略不如 SAC 等 Off-policy 方法丰富
- **超参数敏感**：学习率、熵系数、并行环境数量需要调节

---

## 演化位置

**在策略梯度演化链中的位置**：

```
REINFORCE (1992)
    ↓
Actor-Critic (1983/2000)：引入 Critic 做 baseline
    ↓
A3C (2016)：异步并行，大规模训练
    ↓
A2C (2017)：同步简化版 ← 你在这里
    ↓
PPO (2017)：加 Clip 约束，允许多次复用数据
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象一个公司的决策流程。

- **REINFORCE**：老板一个人做决策，年底看业绩才知道对不对。
- **A3C**：多个分公司各自做决策，做完就向总部汇报，总部随时更新策略。但分公司拿到的是"上周的策略"，可能已经过时了。
- **A2C**：多个分公司同时做决策，做完一起汇报，总部用所有分公司的反馈统一更新策略。每次更新都基于最新信息。
- **PPO**：A2C + 每次更新时限制改动幅度，防止"步子太大扯着蛋"。

**经典实验**：Atari Breakout
- A2C 用 8 个并行环境，在 ~30 分钟内就能达到人类水平
- 比 DQN（需要数小时 + Replay Buffer）快得多
- 但最终得分不如 PPO 稳定

---

## 代码片段

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor_head = nn.Linear(64, action_dim)
        self.critic_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_dist = torch.distributions.Categorical(
            logits=self.actor_head(features))
        value = self.critic_head(features).squeeze(-1)
        return action_dist, value

def a2c_update(model, envs, optimizer, num_steps=5, gamma=0.99, 
               vf_coef=0.5, ent_coef=0.01):
    num_envs = len(envs)
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    
    obs = torch.stack([env.reset() for env in envs])
    
    for _ in range(num_steps):
        dist, value = model(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 并行执行所有环境
        next_obs, reward, done = zip(*[
            envs[i].step(action[i].item()) for i in range(num_envs)
        ])
        
        states.append(obs)
        actions.append(action)
        rewards.append(torch.FloatTensor(reward))
        dones.append(torch.FloatTensor(done))
        log_probs.append(log_prob)
        values.append(value)
        
        obs = torch.stack(next_obs)
    
    # 计算 n-step 回报和优势
    _, last_value = model(obs)
    returns = compute_returns(rewards, dones, last_value, gamma)
    advantages = returns - torch.stack(values)
    
    # 展平
    log_probs = torch.stack(log_probs).view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    
    # 损失计算
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()  # 等价于 (V - R)^2
    entropy = dist.entropy().mean()
    
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
```

---

## 分类与相关算法

- **RL分类**：Model-free, Actor-Critic
- **数据来源**：On-Policy, Online-RL
- **动作空间**：Discrete / Continuous
- **相关算法**：
  - [[02-模块/Actor-Critic/A3C|A3C]]（A2C 的异步前身）
  - [[02-模块/Policy-Based/PPO|PPO]]（A2C + Clip 约束）
  - [[02-模块/Policy-Based/REINFORCE|REINFORCE]]（无 Critic 的基础版本）

---

## 参考资源

- 论文：Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML*.（A3C 论文，A2C 在其中作为 baseline 被提及）
- 博客：OpenAI (2017). "OpenAI Baselines: ACKTR & A2C."
- 代码：[OpenAI Baselines A2C](https://github.com/openai/baselines/tree/master/baselines/a2c)
- 代码：[CleanRL A2C](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c.py)
