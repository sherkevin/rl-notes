---
tags:
  - #model-free
  - #policy-based
  - #on-policy
  - #online-rl
  - #discrete
  - #function-approximation
created: 2026-06-25
---

## [Link] 知识图谱链接

### 相关算法
- [[PPO]]
- [[TRPO]]
- [[A2C]]

### 分类维度
- [[02-模块/Policy-Based/PPO|Policy-Based算法]]
- [[00-索引/按策略类型|On-Policy]]

---

# REINFORCE（REINFORCE — Monte Carlo Policy Gradient）

> 蒙特卡洛策略梯度：用完整轨迹的回报来更新策略参数，是所有策略梯度算法的起点。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：直接对策略参数求梯度，优化期望回报
- **蒙特卡洛估计**：用采样的完整轨迹回报代替真实期望
- **参数化策略** $\pi_\theta$：用神经网络（或线性模型）表示策略，输出动作概率分布

---

## 核心创新

REINFORCE 解决的核心问题：**如何在不依赖值函数的情况下，直接优化策略？**

在 REINFORCE 之前，RL 主要靠 Q-Learning 等值函数方法——先学值函数，再从值函数导出策略。REINFORCE 提出了一个颠覆性的思路：**跳过值函数，直接对策略本身做梯度上升。**

核心思想用一句话概括：**"采样一条轨迹，算出每个动作拿到的总回报，回报高的动作就增大它的概率，回报低的就减小。"**

这是一个极其朴素但数学上严格的思路。Williams (1992) 证明了这个梯度估计是**无偏的**——虽然方差很大，但期望方向是对的。

---

## 算法流程

### 1. 采样完整轨迹

```
对于每个 episode:
    1. 初始化状态 s_0
    2. 对于每个时间步 t = 0, 1, ..., T:
        a. 从策略 π_θ(·|s_t) 采样动作 a_t
        b. 执行 a_t，观察奖励 r_t 和新状态 s_{t+1}
    3. 记录完整轨迹 τ = {(s_0, a_0, r_0), ..., (s_T, a_T, r_T)}
```

### 2. 计算折扣回报

从轨迹末端反向计算每个时刻的回报：

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

```python
# 反向计算折扣回报
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)
```

### 3. 计算策略梯度并更新

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \cdot G_t^{(i)}$$

```python
# 伪代码
log_probs = []
for s, a in zip(states, actions):
    dist = policy(s)
    log_probs.append(dist.log_prob(a))

loss = 0
for t in range(T):
    loss -= log_probs[t] * returns[t]  # 梯度上升 = 最小化负目标

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 4. 重复

用更新后的策略重新采样轨迹。

---

## 关键公式

### 策略梯度定理（REINFORCE 版本）

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

逐项拆解：

1. **$\nabla_\theta \log \pi_\theta(a_t | s_t)$（策略梯度方向）**
   - 含义：对数概率对参数的梯度。它指出"哪个方向能让这个动作的概率变大"。
   - 直觉：想象一个旋钮，转动它能让策略更倾向于在状态 $s_t$ 选择动作 $a_t$。

2. **$G_t$（折扣回报 / 权重）**
   - 含义：从时刻 $t$ 开始，未来所有奖励的加权和。
   - 作用：给梯度一个"权重"——好动作的梯度被放大，坏动作的梯度被缩小甚至反转。

3. **$\mathbb{E}_{\tau \sim \pi_\theta}$（期望）**
   - 含义：对所有可能的轨迹取期望。
   - 实际做法：用采样的 N 条轨迹取平均来近似。

### 带 Baseline 的变体

为了降低方差，通常减去一个 baseline $b(s_t)$（通常是状态值函数 $V(s_t)$）：

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t))$$

- 减去 baseline 不改变期望（数学可证），但能**大幅降低方差**。
- 如果 $b(s_t) = V(s_t)$，则 $G_t - V(s_t)$ 就是 [[01-原子/优势函数|优势函数]] $A(s_t, a_t)$，这正是 REINFORCE with baseline / A2C 的雏形。

---

## 优缺点

### 优点
- **概念简洁**：整个算法就一句话——"好动作概率增大，坏动作概率减小"
- **无偏估计**：梯度估计在数学期望上是正确的
- **适用面广**：不要求动作空间离散，连续动作也能用（只要能计算 $\log \pi$）
- **不需要值函数**：纯策略方法，架构简单

### 缺点
- **方差极大**：必须等整条轨迹跑完才能算回报，一条坏轨迹会严重误导梯度方向
- **样本效率低**：On-policy，每条轨迹只能用一次
- **收敛慢**：高方差意味着需要大量轨迹才能稳定学习
- **信用分配差**：一个动作的好坏要等到 episode 结束才知道，中间无法即时反馈

---

## 演化位置

**在策略梯度演化链中的位置**：

```
REINFORCE (1992) ← 你在这里
    ↓
策略梯度定理 (1999)：给出严格的梯度表达式
    ↓
Actor-Critic：引入值函数做 baseline，降低方差
    ↓
Natural PG (2002)：用 Fisher 信息矩阵做预条件
    ↓
TRPO (2015)：KL 散度约束，保证单调改进
    ↓
PPO (2017)：Clip 机制，简化 TRPO
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象你在学投篮。

- **REINFORCE 的做法**：你投 10 个球（采样一条轨迹），记住每个球的出手角度和力度（动作），然后看看最终进了几个（总回报）。进了的球对应的角度力度，下次多试试；没进的，下次少试试。
- **问题**：如果你这次恰好手感特别好（或特别差），你的"总结"就会被这一次运气带偏。这就是高方差。
- **改进方向**：如果每次出手后马上有个教练告诉你"这球偏了"（即时 TD 反馈），而不是等投完 10 个才总结，学习效率会高很多——这就是 Actor-Critic 的思路。

**经典实验**：CartPole-v1（倒立摆）
- REINFORCE 通常在 500-1000 个 episode 后收敛
- PPO 通常在 100-200 个 episode 就能收敛
- 这体现了纯蒙特卡洛策略梯度与 Actor-Critic 方法的效率差距

---

## 代码片段

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return torch.distributions.Categorical(probs=self.net(x))

def reinforce(env, num_episodes=1000, gamma=0.99, lr=3e-3):
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []
        
        done = False
        while not done:
            dist = policy(torch.FloatTensor(state))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 标准化（减均值除标准差）降低方差
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 策略梯度损失
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G  # 负号 = 梯度上升
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 分类与相关算法

- **RL分类**：Model-free, Policy-Based
- **数据来源**：On-Policy, Online-RL
- **动作空间**：Discrete / Continuous（理论上都支持）
- **相关算法**：
  - [[02-模块/Policy-Based/TRPO|TRPO]]（引入信任域约束）
  - [[02-模块/Policy-Based/PPO|PPO]]（Clip 机制简化 TRPO）
  - [[02-模块/Actor-Critic/A2C|A2C]]（加入值函数 baseline）

---

## 参考资源

- 论文：Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.
- 论文：Sutton et al. (1999). "Policy gradient methods for reinforcement learning with function approximation." *NeurIPS*.
- 代码：[OpenAI Spinning Up VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)（REINFORCE with baseline）
- 教程：[PyTorch REINFORCE](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
