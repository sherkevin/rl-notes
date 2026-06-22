---
tags:
  - module
  - model-free
  - value-based
  - off-policy
  - online-rl
  - discrete
  - function-approximation
created: 2026-06-22
---

# DQN (Deep Q-Network)

> 深度Q网络：将Q-Learning与深度学习结合的开创性算法。2015年DeepMind提出，首次在Atari游戏上达到人类水平。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/Q-Learning]]：基础更新规则
- [[01-原子/经验回放]]：打破数据相关性，实现Off-Policy
- [[01-原子/目标网络]]：稳定训练目标
- **神经网络近似**：用神经网络替代Q表（DQN特有）

---

## 核心创新

### 神经网络近似

将Q-Learning中的Q表替换为神经网络，用函数逼近解决状态空间过大的问题：

$$Q(s, a; \theta) \approx Q^*(s, a)$$

**为什么需要神经网络**：
- Q表只能处理离散状态空间（如棋盘）
- 现实问题状态空间巨大（如图像像素）
- 神经网络可以泛化到未见过的状态

---

## 算法流程

```
初始化：主网络 Q(θ)，目标网络 Q(θ⁻)，经验池 D

对于每个episode：
    初始化状态 s

    对于每个步骤：
        1. 用ε-greedy选择动作 a
        2. 执行 a，观察 r, s'
        3. 存储 (s, a, r, s') 到经验池 D
        4. 从 D 随机采样一批经验
        5. 计算目标：y = r + γ·max Q(s', a'; θ⁻)
        6. 更新主网络：最小化 (y - Q(s, a; θ))²
        7. 每C步同步：θ⁻ ← θ
```

### 损失函数

$$L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]$$

其中目标值：

$$y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^{-})$$

$\theta^{-}$ 是目标网络的参数（定期从主网络复制）

---

## 高估问题与 Double DQN

### 高估问题（Maximization Bias）

`max` 操作天生就是"乐观的"。Q网络因为是近似估计，输出值总会有随机误差（噪声）。

**比喻："过于自信的招聘经理"**

假设招聘经理（Q网络）要评估3个候选人（3个动作），真实能力都是5分，但经理评估有误差，给出 {4.8, 5.2, 4.9}：
- **标准DQN**：执行 `max` 操作，选出5.2分，认为岗位价值就是5.2分
- **问题**：`max` 总是选择被正向误差"眷顾"的动作，系统性高估真实价值

### Double DQN 的解决方案

Double DQN 引入**分权制衡**，将"挑选动作"和"评估动作"分开：

1. **角色分工**：
   - **主网络 `Q_θ`**：动作挑选官，找出Q值最高的动作
   - **目标网络 `Q_{θ'}`**：价值评估官，对选出的动作独立评估

2. **新更新公式**：
   $$y = r + γ \cdot Q_{θ'}(s', \text{argmax}_{a'} Q_θ(s', a'))$$

   - 第一步（挑选）：$a^* = \text{argmax}_{a'} Q_θ(s', a')$ — 主网络选出最佳动作
   - 第二步（评估）：$Q_{θ'}(s', a^*)$ — 目标网络独立评估该动作价值

**效果**：动作必须同时被主网络认为"最好"且被目标网络给出高分，才算真的高价值。两个独立网络都被同一正向误差"眷顾"的概率远小于单个网络，大大缓解了最大化偏差。

---

## 优缺点

### 优点
- ✅ 解决状态空间过大问题
- ✅ 端到端学习（从原始像素到动作）
- ✅ 经验回放提高样本效率
- ✅ 可以处理复杂环境（如图像）

### 缺点
- ❌ 只能处理离散动作
- ❌ Q值高估问题（Double DQN部分缓解）
- ❌ 训练不稳定
- ❌ 超参数敏感

---

## 应用场景

### 适合
- ✅ Atari游戏
- ✅ 棋类游戏（围棋、国际象棋）
- ✅ 任何离散动作空间问题

### 不适合
- ❌ 连续控制问题（用DDPG/SAC/PPO）
- ❌ 需要高频交互的场景

---

## 代码片段

```python
# DQN网络结构
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)

# 计算损失
def compute_loss(batch, main_net, target_net, gamma):
    states, actions, rewards, next_states, dones = batch

    # 当前Q值
    current_q = main_net(states).gather(1, actions)

    # 目标Q值
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target_q = rewards + gamma * next_q * (1 - dones)

    # Huber损失
    loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
    return loss
```

---

## 演化位置

**在 Value-Based 演化链中的位置**：

```
Q-Learning (表格)
    ↓
DQN (神经网络) ← **你在这里**
    ↓
DDQN (解决高估)
    ↓
Dueling DQN (网络结构改进)
    ↓
Rainbow DQN (多项改进集成)
    ↓
CQL (离线版本)
```

详见 [[03-流程/Value-Based演化史]]

---

## 参考资源

- 论文：Mnih et al. "Human-level control through deep reinforcement learning" (Nature 2015)
