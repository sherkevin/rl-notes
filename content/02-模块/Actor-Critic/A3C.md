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
- [[A2C]]
- [[PPO]]

### 分类维度
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|On-Policy]]

---

# Asynchronous Advantage Actor-Critic (A3C)

> 异步优势演员-评论家：用多个异步 worker 并行探索环境，实现大规模分布式强化学习训练。深度 RL 并行化的开创性工作。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：Actor 直接优化策略参数
- [[01-原子/优势函数]]：$A(s,a) = Q(s,a) - V(s)$ 降低策略梯度方差
- **Actor-Critic 架构**：Actor 学策略，Critic 学值函数 $V(s)$
- **异步并行训练**：多个 worker 独立运行，各自计算梯度并异步更新全局网络
- [[01-原子/GAE]]：广义优势估计（n-step 回报的特例）

---

## 核心创新

A3C 解决的核心问题：**如何在不依赖 GPU 集群和 Replay Buffer 的情况下，高效训练深度 RL 模型？**

在 A3C 之前，深度 RL（如 DQN）的标准做法是：单 agent + GPU + 大容量 Replay Buffer。这需要昂贵的硬件和复杂的数据管理。

**核心思想**（Mnih et al., 2016）：用多个 CPU worker 并行跑多个环境副本，每个 worker 维护一份全局网络的本地拷贝，独立与环境交互、计算梯度，然后**异步**地将梯度推送给全局网络。

三个关键创新：
1. **异步更新**：不需要等所有 worker 完成，每个 worker 算完梯度就立刻推送给全局网络
2. **不需要 Replay Buffer**：多 worker 并行本身就提供了数据多样性（等价于 decorrelating samples）
3. **CPU 即可训练**：不用 GPU，在普通多核 CPU 上就能高效训练

---

## 算法流程

### 1. 初始化

```
1. 创建全局网络 (Actor π_θ, Critic V_φ)，参数为 θ, φ
2. 创建 N 个 worker，每个 worker 有自己的：
   - 本地网络拷贝 (π_{θ'}, V_{φ'})
   - 独立的环境副本
```

### 2. 每个 Worker 独立执行（异步）

```
Worker i 的循环：
    1. 从全局网络同步参数：θ' ← θ, φ' ← φ
    2. 用本地网络收集 n 步数据：
       for t = 1 to n:
           a_t ~ π_{θ'}(·|s_t)
           执行 a_t，得到 r_t, s_{t+1}, done_t
    3. 计算 n-step 回报：
       R = { V_{φ'}(s_{t+n})   如果未结束
           { r_t               如果 episode 结束
       反向计算：R_t = r_t + γ · R_{t+1}
    4. 计算优势：A_t = R_t - V_{φ'}(s_t)
    5. 累积梯度（对本地网络参数求导）：
       dθ' += ∇_{θ'} log π_{θ'}(a_t|s_t) · A_t
       dφ' += ∇_{φ'} (R_t - V_{φ'}(s_t))^2
       dθ' += β · ∇_{θ'} H(π_{θ'}(·|s_t))    // 熵正则
    6. 将累积梯度异步推送到全局网络：
       θ ← θ + α · dθ'    // 无锁更新
       φ ← φ + α · dφ'
    7. 回到步骤 1
```

### 3. 训练终止

当全局网络达到预定性能或训练步数时，停止所有 worker。

---

## 关键公式

### 策略梯度（每个 Worker 计算）

$$d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \cdot A_t + \beta \nabla_{\theta'} H(\pi_{\theta'}(\cdot|s_t))$$

- 第一项：增大优势为正的动作概率
- 第二项：熵正则，鼓励探索，防止策略过早收敛

### 值函数梯度

$$d\phi \leftarrow d\phi + \nabla_{\phi'} (R_t - V_{\phi'}(s_t))^2$$

### n-step 优势估计

$$A_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_{\phi'}(s_{t+n}) - V_{\phi'}(s_t)$$

- $n$ 通常取 5-20（平衡偏差-方差）
- 当 $n \to \infty$ 时，退化为蒙特卡洛估计（REINFORCE）
- 当 $n = 1$ 时，退化为 TD(0)（高偏差、低方差）

### 全局参数异步更新

$$\theta \leftarrow \theta + \alpha \cdot d\theta_i$$

- 每个 worker $i$ 独立计算 $d\theta_i$，然后**异步**地加到全局参数上
- **没有锁、没有同步屏障**——这意味着 worker 可能用的是"稍旧的"全局参数
- 实践表明，这种"staleness"对训练影响很小，但大幅提升了吞吐量

---

## 架构细节

### 网络共享设计

A3C 通常使用**共享特征提取层 + 分离的输出头**：

```
输入: 状态 s
    ↓
[共享卷积/全连接层] → 特征向量 f
    ↓                    ↓
[Actor Head]        [Critic Head]
    ↓                    ↓
π(a|s)              V(s)
(动作概率分布)       (状态价值)
```

- 共享层学习通用的状态表示
- 分离输出头避免任务冲突

### Hogwild! 异步更新

A3C 使用 Dean et al. (2011) 提出的 **Hogwild!** 方法进行异步 SGD：
- 多个 worker 同时读写共享参数，不加锁
- 理论上可能产生"过期梯度"（stale gradients）
- 实践中，由于 RL 本身的随机性，这种近似是可接受的

---

## 优缺点

### 优点
- **训练速度快**：多 worker 并行 = 数据收集速度线性增长
- **不需要 GPU**：纯 CPU 训练，降低硬件门槛
- **不需要 Replay Buffer**：多 worker 天然提供数据多样性
- **探索性好**：每个 worker 的策略略有不同（参数异步更新导致），天然鼓励多样化探索
- **可扩展性强**：轻松扩展到数十个 worker

### 缺点
- **异步导致参数过期**：Worker 使用的参数可能已经过时，影响梯度质量
- **GPU 利用率低**：小批次异步更新不适合 GPU 的并行计算模式
- **调参复杂**：Worker 数量、学习率衰减策略、n-step 长度等需要调节
- **复现困难**：异步执行导致训练过程不确定性强
- **被 A2C 取代**：同步版 A2C 更简单、更稳定、GPU 利用率更高

---

## 演化位置

**在策略梯度演化链中的位置**：

```
REINFORCE (1992)
    ↓
Actor-Critic (1983/2000)
    ↓
Natural PG / TRPO (2002/2015)
    ↓
A3C (2016)：异步并行 ← 你在这里
    ↓
A2C (2017)：同步简化版
    ↓
IMPALA (2018)：大规模分布式，解耦 actor 和 learner
    ↓
PPO (2017)：加 Clip 约束
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象一个考古发掘团队。

- **DQN（单 agent）**：一个人拿着铲子挖，挖到的东西都存进仓库（Replay Buffer），然后回仓库翻旧货学习。
- **A3C（异步多 agent）**：一个团队分散在不同区域同时挖，每人挖到一点就通过对讲机向总部汇报，总部实时更新"地图"。每个人手里拿的地图可能不是最新的（异步），但总体方向是对的。
- **A2C（同步多 agent）**：团队分散挖掘，但约定每隔一段时间统一汇报，总部用所有信息更新地图后再分发。每个人永远有最新地图。

**经典实验**：Atari 游戏
- A3C 在 Atari 上首次证明：深度 RL 可以纯 CPU 训练
- 用 16 个 CPU worker 在 24 小时内超过 DQN（GPU 训练数天）的效果
- 但后来被 A2C + GPU 以更少的时间超越

---

## 代码片段

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SharedModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        # 共享梯度缓冲区
        for p in self.parameters():
            p.share_memory_()

def worker(global_model, optimizer, env_fn, worker_id, n_steps=5, gamma=0.99):
    env = env_fn()
    local_model = SharedModel(env.observation_space.shape[0], env.action_space.n)
    
    state = env.reset()
    while True:
        # 1. 同步全局参数
        local_model.load_state_dict(global_model.state_dict())
        
        log_probs, values, rewards, entropies = [], [], [], []
        
        # 2. 收集 n 步数据
        for _ in range(n_steps):
            state_t = torch.FloatTensor(state)
            features = local_model.shared(state_t)
            logits = local_model.actor(features)
            value = local_model.critic(features)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            entropies.append(dist.entropy())
            
            state = next_state
            if done:
                state = env.reset()
                break
        
        # 3. 计算 n-step 回报
        R = torch.zeros(1) if done else local_model.critic(
            local_model.shared(torch.FloatTensor(state))).detach()
        
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns).detach()
        
        # 4. 计算损失
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        advantages = returns - values.detach()
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -torch.stack(entropies).mean()
        
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        # 5. 计算梯度并推送到全局模型
        optimizer.zero_grad()
        loss.backward()
        # 将本地梯度复制到全局模型
        for local_p, global_p in zip(local_model.parameters(), 
                                      global_model.parameters()):
            global_p._grad = local_p.grad
        optimizer.step()

# 启动
# global_model = SharedModel(state_dim, action_dim).share_memory()
# optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
# for i in range(num_workers):
#     p = mp.Process(target=worker, args=(global_model, optimizer, env_fn, i))
#     p.start()
```

---

## 分类与相关算法

- **RL分类**：Model-free, Actor-Critic
- **数据来源**：On-Policy, Online-RL
- **动作空间**：Discrete / Continuous
- **并行架构**：异步多 worker
- **相关算法**：
  - [[02-模块/Actor-Critic/A2C|A2C]]（同步简化版，推荐替代 A3C）
  - [[02-模块/Policy-Based/PPO|PPO]]（A2C + Clip 约束）
  - [[02-模块/Policy-Based/TRPO|TRPO]]（信任域约束）
  - **IMPALA**（大规模分布式，解耦 actor/learner）

---

## 参考资源

- 论文：Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML*.
- 论文：Dean et al. (2011). "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent." *NeurIPS*.
- 代码：[OpenAI Universe Starter Agent](https://github.com/openai/universe-starter-agent)（A3C 参考实现）
- 代码：[ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)（高质量 PyTorch 实现）
- 教程：[Morvan Zhou A3C Tutorial](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
