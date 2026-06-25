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
- [[DDPG]]
- [[SAC]]

### 分类维度
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|Off-Policy]]
- [[00-索引/按动作空间|Continuous-Actions]]

---

# Twin Delayed DDPG (TD3)

> 双延迟 DDPG：针对 DDPG 的三大改进——双 Critic 解决 Q 值高估、延迟更新稳定训练、目标策略平滑防止过拟合。连续控制领域的强基线算法。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：Actor 通过确定性策略梯度优化
- [[01-原子/Q-Learning]]：Critic 用 TD 更新（取双 Critic 最小值）
- [[01-原子/经验回放]]：Replay Buffer 存储历史转移
- [[01-原子/目标网络]]：Soft update 稳定训练
- **Clipped Double Q-Learning**：双 Critic + 取最小值，解决高估
- **延迟策略更新**：Critic 更新多次后 Actor 才更新一次
- **目标策略平滑**：给目标动作加噪声，防止 Critic 在狭窄峰值上过拟合

---

## 核心创新

TD3 解决的核心问题：**DDPG 的 Q 值高估问题有多严重，以及如何系统性地解决？**

Fujimoto et al. (2018) 发现 DDPG 的 Q 值严重高估——Critic 对动作的评价远高于真实回报。这导致 Actor 被"骗"去追逐虚假的高分动作，最终策略性能下降。

**核心思想**：三个相互关联的技巧共同解决高估和不稳定问题：

1. **Clipped Double Q-Learning**（双 Critic）：训练两个独立的 Critic，计算目标 Q 值时取两者的**最小值**——"宁可信其低，不可信其高"
2. **Delayed Policy Updates**（延迟更新）：Critic 先多学几步稳定下来，Actor 再跟着更新——"先让评价标准靠谱，再按标准改进动作"
3. **Target Policy Smoothing**（目标策略平滑）：给目标动作加噪声——"别让 Critic 记住某个精确动作的高分，应该学一片区域都不错"

---

## 算法流程

### 1. 初始化

```
1. Actor 网络 μ_θ
2. 两个 Critic 网络 Q_{φ1}, Q_{φ2}
3. 目标网络：μ_{θ'}, Q_{φ1'}, Q_{φ2'}
4. Replay Buffer D
```

### 2. 数据收集

```
对于每个时间步 t:
    a_t = μ_θ(s_t) + ε     // ε ~ N(0, σ_exploration^2)
    执行 a_t，得到 r_t, s_{t+1}
    存储 (s_t, a_t, r_t, s_{t+1}) 到 D
```

### 3. 网络更新（每步都做）

```
从 D 中采样 Mini-batch {(s_i, a_i, r_i, s'_i)}：

(1) 计算目标 Q 值（目标策略平滑 + 双 Critic 取最小）：
    a'_i = μ_{θ'}(s'_i) + clip(ε', -c, c)    // 目标动作加裁剪噪声
    ε' ~ N(0, σ_target^2)
    y_i = r_i + γ · min(Q_{φ1'}(s'_i, a'_i), Q_{φ2'}(s'_i, a'_i))

(2) 更新两个 Critic：
    L_{Q1} = MSE(Q_{φ1}(s_i, a_i), y_i)
    L_{Q2} = MSE(Q_{φ2}(s_i, a_i), y_i)

(3) 延迟更新 Actor（每 d 步做一次，d 通常 = 2）：
    if total_updates % d == 0:
        L_μ = -(1/N) Σ Q_{φ1}(s_i, μ_θ(s_i))    // 只用第一个 Critic

(4) Soft Update 目标网络（每 d 步做一次）：
    θ' ← τθ + (1-τ)θ'
    φ1' ← τφ1 + (1-τ)φ1'
    φ2' ← τφ2 + (1-τ)φ2'
```

---

## 关键公式

### (1) Clipped Double Q-Learning（目标 Q 值计算）

$$y = r + \gamma \cdot \min_{j=1,2} Q_{\phi'_j}(s', \mu_{\theta'}(s'))$$

**对比 DDPG**：
$$y_{\text{DDPG}} = r + \gamma \cdot Q_{\phi'}(s', \mu_{\theta'}(s'))$$

- DDPG 只有一个 Critic，容易高估
- TD3 有两个 Critic，取最小值 = 保守估计
- **直觉**：两个评委打分，取低分——这样 Actor 不会被过分乐观的评价误导

### (2) 目标策略平滑（Target Policy Smoothing）

$$a' = \mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

- $\sigma$ 通常取 0.2，$c$ 通常取 0.5
- **为什么加噪声？** 防止 Critic 在某个精确动作上过拟合出一个尖锐的 Q 值峰值
- **直觉**：就像学开车，不是说"方向盘恰好 37.2 度就是满分"，而是"35-40 度这个范围都不错"。加噪声让 Critic 学到平滑的价值函数
- **Clip 的作用**：噪声不能太大，否则目标动作偏离太远，失去意义

### (3) Actor 延迟更新

$$\text{Actor 每 } d \text{ 步 Critic 更新后才更新一次}$$

- $d$ 通常取 2
- **为什么延迟？** Critic 需要更充分的学习才能给出可靠评价。如果 Actor 在 Critic 还没学好的时候就跟着改，容易被带偏
- **类比**：先让教练多看几场比赛、形成稳定的评价标准，运动员再按教练的标准调整动作

### (4) Actor Loss（与 DDPG 相同）

$$L(\theta) = -\mathbb{E}_{s \sim D} [Q_{\phi_1}(s, \mu_\theta(s))]$$

- 只用第一个 Critic 计算 Actor 梯度（不需要两个都用）
- 梯度方向：让 Actor 输出的动作使 Q 值变大

---

## 三大改进详解

### 改进一：双 Critic 解决高估

| | DDPG | TD3 |
|---|---|---|
| Critic 数量 | 1 | 2 |
| 目标 Q 值 | $Q(s',a')$ | $\min(Q_1(s',a'), Q_2(s',a'))$ |
| 高估程度 | 严重 | 大幅缓解 |
| 训练稳定性 | 差 | 好 |

**为什么取最小值有效？** 两个 Critic 独立训练，各自的高估方向不同。取最小值相当于"两个乐观的评委中，听较保守的那个"。数学上可以证明，$\min$ 操作给出了一个低估的 Q 值（下界），这比高估更安全——低估只是让你"过于谨慎"，高估会让你"盲目自信"。

### 改进二：延迟更新

| | DDPG | TD3 |
|---|---|---|
| Actor 更新频率 | 每步 | 每 2 步 |
| Critic 更新频率 | 每步 | 每步 |
| 效果 | Actor 被不靠谱的 Critic 带偏 | Critic 先稳定，Actor 再跟上 |

### 改进三：目标策略平滑

| | DDPG | TD3 |
|---|---|---|
| 目标动作 | $a' = \mu_{\theta'}(s')$ | $a' = \mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c)$ |
| Q 值函数 | 可能有尖锐峰值 | 平滑 |
| 过拟合风险 | 高 | 低 |

---

## 优缺点

### 优点
- **高估问题大幅缓解**：双 Critic + 取最小值，比 DDPG 保守得多
- **训练更稳定**：延迟更新 + 目标策略平滑共同作用
- **性能显著提升**：在 MuJoCo 上通常比 DDPG 好 2-3 倍
- **实现不太复杂**：相比 DDPG 只增加了约 20 行代码
- **强基线**：连续控制任务中常用的 baseline 算法

### 缺点
- **计算量翻倍**：两个 Critic 意味着前向传播和梯度计算量翻倍
- **仍然不如 SAC**：SAC 的最大熵框架提供更好的探索机制
- **探索方式粗糙**：仍然靠加噪声探索，不如 SAC 的熵正则优雅
- **超参数仍有影响**：延迟步数 $d$、噪声参数 $\sigma$ 和 $c$ 需要调节
- **纯连续动作**：不适合离散动作空间

---

## 演化位置

**在连续控制演化链中的位置**：

```
DPG (2014)：确定性策略梯度定理
    ↓
DDPG (2016)：深度版本
    ↓
TD3 (2018)：双 Critic + 延迟更新 + 目标策略平滑 ← 你在这里
    ↓
SAC (2018)：最大熵 + 自适应温度
    ↓
REDQ (2021)：更多 Critic + 更激进的更新
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象你在学做一道菜。

- **DDPG**：一个师傅教你，他有时候过于乐观（"你这道菜做得快赶上大师了！"），你就信了，结果比赛时翻车。
- **TD3 的三个改进**：
  1. **双师傅（双 Critic）**：请两个师傅评价，听较严格的那个——这样你不会过于自信
  2. **延迟改进（延迟更新）**：师傅们先看几天你的操作，形成稳定标准后，你再按标准改。而不是师傅每天标准都在变，你跟着瞎改
  3. **模糊标准（目标策略平滑）**：师傅不是说"盐恰好 3.7 克"，而是说"3-4 克都行"——这样你不会对某个精确数值过拟合

**经典实验**：MuJoCo Walker2d-v2
- DDPG：~1500 分（高估严重，训练不稳定）
- TD3：~4000 分（稳定提升 2-3 倍）
- SAC：~5000 分（略优于 TD3）

---

## 代码片段

```python
import torch
import torch.nn.functional as F
from copy import deepcopy

class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, policy_noise=0.2, 
                 noise_clip=0.5, policy_delay=2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        
        self.actor_target = deepcopy(self.actor)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0
    
    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # --- 计算目标 Q 值（三大改进集中在这里）---
        with torch.no_grad():
            # 改进3：目标策略平滑——给目标动作加裁剪噪声
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise
                          ).clamp(-self.max_action, self.max_action)
            
            # 改进1：双 Critic 取最小值
            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(q1_next, q2_next)
        
        # 更新两个 Critic
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        # critic_optimizer.zero_grad(); critic_loss.backward(); critic_optimizer.step()
        
        # --- 改进2：延迟策略更新 ---
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            # Actor 更新
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            # actor_optimizer.zero_grad(); actor_loss.backward(); actor_optimizer.step()
            
            # Soft Update 目标网络
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

---

## TD3 vs DDPG vs SAC 对比

| 特性 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| Critic 数量 | 1 | 2 | 2 |
| 高估问题 | 严重 | 缓解 | 缓解 |
| 探索方式 | 加噪声 | 加噪声 | 最大熵 |
| 策略类型 | 确定性 | 确定性 | 随机性 |
| 延迟更新 | 否 | 是 | 否 |
| 目标平滑 | 否 | 是 | 否 |
| 超参数敏感度 | 高 | 中 | 低 |
| 综合性能 | 较低 | 中 | 高 |

---

## 分类与相关算法

- **RL分类**：Model-free, Actor-Critic, Off-Policy
- **数据来源**：Off-Policy, Online-RL
- **动作空间**：Continuous
- **策略类型**：确定性策略（Deterministic Policy）
- **相关算法**：
  - [[02-模块/Actor-Critic/DDPG|DDPG]]（TD3 的前身，存在高估问题）
  - [[02-模块/Actor-Critic/SAC|SAC]]（最大熵版本，探索更好，SOTA）
  - [[02-模块/Value-Based/DQN|DQN]]（Clipped Double Q-Learning 的来源）

---

## 参考资源

- 论文：Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*.
- 代码：[作者原版实现](https://github.com/sfujim/TD3)
- 代码：[OpenAI Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- 代码：[Stable Baselines3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- 代码：[CleanRL TD3](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)
