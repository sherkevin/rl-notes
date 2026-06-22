---
tags:
  - module
  - model-free
  - policy-based
  - on-policy
  - online-rl
  - continuous
  - discrete
  - function-approximation
created: 2026-06-22
---

# PPO (Proximal Policy Optimization)

> 近端策略优化：通过Clip机制限制策略更新幅度，实现稳定高效的策略学习。目前应用最广泛的RL算法之一。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：直接优化策略参数
- [[01-原子/优势函数]]：降低[[01-原子/策略梯度|策略梯度]]方差
- [[01-原子/GAE]]：广义优势估计，平衡偏差-方差
- [[01-原子/Clip机制]]：限制策略更新幅度
- **Actor-Critic架构**：Actor学习策略，Critic学习价值函数

---

## 核心创新

PPO的主要目标是解决传统[[01-原子/策略梯度|策略梯度]]算法的**学习过程不稳定**问题。

**核心思想**：在学习中，既要大胆进步，又要小心翼翼，别把已经学会的好东西给忘了。

通过**[[01-原子/Clip机制|Clip机制]]**（详见 [[01-原子/Clip机制]]），PPO确保每次策略更新都在"安全的"、"可信的"范围内进行小步慢跑，而不是大步乱跳。

---

## 算法流程

### 1. 数据收集

使用当前策略 $\pi_{\theta_\text{old}}$ 与环境交互，收集轨迹数据：

```
对于每个episode:
    1. 初始化状态 s_0
    2. 对于每个时间步 t:
        a. 从 π_{θ_old}(·|s_t) 采样动作 a_t
        b. 执行 a_t，观察奖励 r_t 和新状态 s_{t+1}
    3. 存储轨迹 {(s_t, a_t, r_t)}
```

### 2. 计算优势

使用GAE（详见 [[01-原子/GAE]]）计算优势估计：

$$A^\text{[[01-原子/GAE|GAE]]}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

### 3. 更新策略

使用Clip目标函数（详见 [[01-原子/Clip机制]]）：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$

### 4. 更新价值函数

$$L^{VF}(\phi) = \mathbb{E}_t\left[(V_\phi(s_t) - \hat{R}_t)^2\right]$$

其中 $\hat{R}_t = \sum_{l=0}^\infty \gamma^l r_{t+l}$

### 5. 重复

丢弃旧数据，用更新后的策略 $\pi_\theta$ 重新收集数据。

---

## 关键参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $\epsilon$ | 0.1 或 0.2 | Clip范围，控制策略更新幅度 |
| $\gamma$ | 0.99 | 折扣因子 |
| $\lambda$ | 0.95 | GAE参数，控制偏差-方差权衡 |
| 学习率 | $3\times 10^{-4}$ | Adam优化器学习率 |
| Epoch数 | 10 | 每批数据的训练轮数 |
| Batch size | 64 | 每批数据的大小 |

---

## 代码片段

```python
import torch
import torch.nn.functional as F

# 假设 actor, critic, old_actor 已定义

# 1. 收集数据（伪代码）
# states, actions, rewards, dones = collect_data(old_actor, env, num_steps)

# 2. 计算优势（GAE）
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)

# 3. PPO更新
def ppo_update(actor, critic, old_actor, states, actions, advantages, returns):
    # 计算概率比
    pi = actor(states, actions)
    old_pi = old_actor(states, actions).detach()
    ratio = torch.exp(pi - old_pi)
    
    # PPO-Clip目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 价值函数损失
    values = critic(states).squeeze()
    value_loss = F.mse_loss(values, returns)
    
    # 总损失
    loss = policy_loss + 0.5 * value_loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 优缺点

### 优点
- ✅ **稳定性好**：Clip机制防止策略更新过大
- ✅ **样本效率较高**：On-policy但可多次使用同一批数据
- ✅ **实现简单**：比TRPO简单得多
- ✅ **通用性强**：适用于离散和连续动作空间
- ✅ **调参容易**：默认参数在大多数任务上表现良好

### 缺点
- ❌ **样本效率不如Off-policy**：每批数据只能用几次
- ❌ **On-policy限制**：不能使用历史数据
- ❌ **计算成本**：需要同时训练Actor和Critic
- ❌ **Critic误差影响**：Critic估计不准会影响策略更新

---

## 应用场景

### 适合
- ✅ 机器人控制（MuJoCo、PyBullet）
- ✅ 游戏AI（Atari、Dota 2）
- ✅ 自动驾驶
- ✅ LLM对齐（RLHF）

### 不适合
- ❌ 样本收集昂贵的场景（用Off-policy方法）
- ❌ 需要离线学习的场景（用Offline RL）
- ❌ 多智能体协作（用MAPPO）

---

## 演化位置

**在 Policy-Based 演化链中的位置**：

```
策略梯度 (1999)
    ↓
REINFORCE (1992)
    ↓
Actor-Critic (1999)
    ↓
TRPO (2015)：KL散度硬约束
    ↓
PPO (2017)：Clip软约束 ← **你在这里**
    ↓
GRPO (2024)：去Critic，组内标准化
    ↓
DAPO (2025)：改进Clip，动态采样
```

详见 [[03-流程/Policy-AC演化史]]

---

## 相关算法

- [[02-模块/Policy-Based/REINFORCE]]：基础[[01-原子/策略梯度|策略梯度]]
- [[02-模块/Policy-Based/TRPO]]：PPO的前身，使用KL散度约束
- [[02-模块/Policy-Based/GRPO]]：PPO的简化版，去掉了Critic
- [[02-模块/Policy-Based/DAPO]]：PPO的改进版，用于LLM对齐
- [[02-模块/Actor-Critic/A2C]]：PPO的同步版本
- [[02-模块/Actor-Critic/A3C]]：PPO的异步版本

---

## 参考资源

- 论文：Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- 代码：[OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- 代码：[CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
