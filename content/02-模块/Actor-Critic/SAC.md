---
tags:
  - #model-free
  - #actor-critic
  - #off-policy
  - #online-rl
  - #continuous
  - #function-approximation
  - off-policy
  - Online
  - Model-free
---

## [Link] 知识图谱链接

### 相关算法
- [[TD3]]
- [[DDPG]]
- [[PPO]]

### 分类维度
- [[02-模块/Actor-Critic/]]
- [[Off-Policy]]
- 

## 1. 核心概述 (Executive Summary)

**Soft Actor-Critic (SAC)** 是一种**面向连续动作空间**的深度强化学习算法。

- **类型：** Off-policy (离线策略)
- **架构：** Actor-Critic (演员-评论家)
- **核心理论：** Maximum Entropy Reinforcement Learning (最大熵强化学习)
- **地位：** 目前是 Model-free RL（无模型强化学习）中的 SOTA（State-of-the-art）基准算法之一，以**高样本效率**和**对超参数由于稳健**著称。
    

---

## 2. 核心理论：最大熵强化学习 (The Essence)

SAC 与传统 RL（如 [[DDPG]], [[PPO]]）最大的区别在于它的**目标函数**。

- 传统 RL 目标： 寻找一个策略 $\pi$，最大化累积奖励的期望值。
    
    $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}[r(s_t, a_t)]$$
    
    缺点： 容易陷入局部最优，探索性（Exploration）不足，往往只收敛到一个确定性的动作。
    
- SAC (最大熵 RL) 目标： 最大化累积奖励，同时最大化策略的熵（Entropy）。
    
    $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}[r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$
    
    - $H(\pi(\cdot|s_t))$ [[基本缩写含义#H pi cdot s_t 具体指的是什么？]]是熵，衡量策略的随机性。
    - $\alpha$ 是温度系数（Temperature），控制奖励和熵之间的权衡。

**为什么这么做？**
1. **鼓励探索：** 如果多个动作都能获得相似的高分，SAC 会倾向于给它们分配相等的概率，而不是只选某一个。
2. **鲁棒性：** 策略不会过早收敛到某一个具体的点，这使得它面对环境扰动时更具鲁棒性。

---

## 3. 模型架构 (Model Architecture)
现代 SAC（通常指 SAC-v2，去掉了 Value Network）主要包含以下神经网络：
### A. Actor Network (策略网络, $\pi_\phi$)
- **输入：** 状态 $s$。
- **输出：** 高斯分布的参数——均值 $\mu$ 和 标准差 $\log \sigma$。
- **动作生成：** 通过重参数化技巧（Reparameterization Trick）从分布中采样动作，并经过 $\tanh$ 激活函数将动作限制在 $[-1, 1]$ 之间。

### B. Critic Network (价值网络, $Q_\theta$)
- **输入：** 状态 $s$ 和 动作 $a$。
- **输出：** Q值（标量）。
- **Clipt Double Q-Learning：** 为了解决 Q 值高估问题，SAC 使用**两个**独立的 Critic 网络 ($Q_{\theta_1}, Q_{\theta_2}$)，计算目标时取两者中的**最小值**。

### C. Target Critic Networks ($Q_{\theta_{target}}$)
- 为了训练稳定性，维护两个目标网络，其参数是主 Critic 网络的指数移动平均（EMA/Soft Update）。

 **A. “主 Critic” 指的是谁？**
SAC 有**两个**正在被梯度下降训练的 Critic 网络，通常叫 $Q_{\theta_1}$ 和 $Q_{\theta_2}$。
- **“主 Critic”** 指的就是这**两个**正在实时学习、参数不断更新的网络。
- 它们是干活的主力，每次迭代参数都会变。

**B. 什么是 Target Critic (目标网络)？**
为了训练稳定，每个主 Critic 都有一个对应的**影子分身**，叫 Target Critic ($Q_{\theta_{target1}}, Q_{\theta_{target2}}$)。
- **计算 Target Q 值时（也就是计算 $y = r + \dots$ 这一步），我们不用主网络，而是用这些影子分身来算。**
- 为什么？因为如果 $y$ 里的参数也在变，那这就变成了“左脚踩右脚上天”，训练会震荡不收敛。我们需要目标 $y$ 暂时是固定的（或者变化很慢的）。

**C. 指数移动平均 (EMA / Soft Update) 是怎么做的？**
在传统的 DQN 里，Target 网络是每隔几千步直接把主网络的参数**硬拷贝（Hard Copy）** 过来。
但在 SAC（以及 DDPG）里，使用 Soft Update，即每一步都稍微更新一点点。
公式：
$$\theta_{target} \leftarrow \tau \cdot \theta_{main} + (1 - \tau) \cdot \theta_{target}$$
- $\theta_{main}$：主 Critic 的参数（最新的）。
- $\theta_{target}$：Target Critic 的参数（旧的）。
- $\tau$ (Tau)：软更新系数，通常很小（例如 0.005）。

人话解释：
这就好比主 Critic 是现在的你（每天都在变），Target Critic 是你的照片。
- **Hard Copy:** 每过一年拍一张新照片。
- **Soft Update:** 每一天都在 PS 修图，把昨天的照片往今天的样子修 **0.5%**。这样照片虽然永远滞后于真人，但它变化非常平滑，不会突变，这对数学上的收敛非常重要。

## 4. 关键公式与 Loss 函数 (The Math)
SAC 的更新涉及三个部分：Critic 更新、Actor 更新、以及 Alpha (温度) 自动调节。
### (1) Critic Loss (Q网络更新)
目的： 让 Q 网络估算的价值，尽可能接近真实的“环境反馈 + 未来预期”。
这是一个回归问题（Regression），通常使用均方误差（MSE）。

#### 核心公式

$$L(\theta) = \frac{1}{N} \sum (Q_\theta(s, a) - y)^2$$

其中，目标值 (Target) $y$ 的计算是最关键的：

$$y = r(s, a) + \gamma \left( \min_{j=1,2} Q_{\theta_{target}, j}(s', a') - \alpha \ln \pi_\phi(a'|s') \right)$$

#### 逐项拆解
1. **$Q_\theta(s, a)$ (当前预测值)**
    - **含义：** 主 Critic 网络在当前状态 $s$ 下，对当前动作 $a$ 给出的打分。
    - **来源：** 直接前向传播（Forward Pass）。
        
2. **$y$ (TD Target / 时序差分目标)**
    - **含义：** 我们认为的“标准答案”。它由两部分组成：眼前的奖励 + 未来的价值。

3. **$r(s, a)$ (即时奖励)**    
    - **含义：** 环境给的反馈（Scale Reward）。
    - **作用：** 这是最真实的信号，是整个学习的基础。
        
4. **$\gamma$ (Gamma, 折扣因子)**
    - **含义：** 0~1 之间的数（通常 0.99）。
    - **作用：** 决定 Agent 即使行乐还是目光长远。0.99 意味着第 100 步后的奖励对现在也很重要。
        
5. **$s', a'$ (下一时刻的状态和动作)**
    - **$s'$：** 从 Replay Buffer 里取出的下一个状态。
    - **$a'$：** **注意！** 这个动作不是 Buffer 里存的历史动作，而是**用当前的 Actor 网络基于 $s'$ 现算出来的（采样出来的）**。这是为了评估“如果按照现在的策略继续走，未来能得多少分”。
        
6. **$\min_{j=1,2} Q_{\theta_{target}, j}(s', a')$ (Clipped Double Q-Learning)**
    - **含义：** 用两个 Target Critic 网络分别算 $Q$ 值，然后**取较小的那一个**。
    - **作用：** **防止高估**。Q-Learning 容易盲目乐观，取最小值是泼冷水，让 Agent 保守一点，训练更稳。
        
7. **$-\alpha \ln \pi_\phi(a'|s')$ (熵奖励项 / Soft Term)**
    - **核心中的核心！** 这是 SAC 区别于 DDPG 的地方。
    - **$\ln \pi$：** 动作的对数概率（Log Probability）。因为概率 < 1，所以 $\ln \pi$ 是负数。
    - **$-\ln \pi$：** 变成了正数。概率越小（越不确定），这个值越大（熵越大）。
    - **含义：** 在计算未来价值时，不仅看 $Q$ 值，还要看“未来的这个动作够不够随机”。**如果未来能保持高度随机性（高熵），我们就给它加分。**$$L(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim D} [ (Q_{\theta_i}(s, a) - y)^2 ]$$
### (2) Actor Loss (策略网络更新)
**目的：** 调整策略网络的参数，使得选出的动作既能拿高分（高 Q），又能保持随机（高熵）。

#### 核心公式

$$L(\phi) = \mathbb{E}_{s \sim D, \epsilon \sim \mathcal{N}} [ \alpha \ln \pi_\phi(a|s) - Q_\theta(s, a) ]$$

这里使用了**重参数化技巧 (Reparameterization Trick)**，即 $a = f_\phi(\epsilon, s) = \tanh(\mu_\phi(s) + \sigma_\phi(s) \cdot \epsilon)$。

#### 逐项拆解

1. **$\epsilon \sim \mathcal{N}(0, I)$ (高斯噪声)**
    - **含义：** 从标准正态分布采样的噪声。
    - **作用：** 它是随机性的源头。因为要对神经网络求导，不能直接在网络中间搞随机采样（那样梯度会断掉），所以把随机性作为“输入”乘进去。这就是重参数化。
        
2. **$a$ (当前的动作)**
    - **含义：** Actor 网络基于 $s$ 和噪声 $\epsilon$ 算出来的动作。
    - **注意：** 这里的动作带有梯度（Gradient），可以反向传播修参数。
        
3. **$Q_\theta(s, a)$ (Q 值评价)**
    - **含义：** Critic 觉得 Actor 选的这个动作 $a$ 怎么样。
    - **符号：** 公式里是 **减号 ($-$)**。因为我们要**最大化** Q 值，也就是**最小化** $-Q$。
        
4. **$\alpha \ln \pi_\phi(a|s)$ (熵正则项)**
    - **含义：** 当前策略选这个动作的概率的对数。
    - **作用：** 我们希望熵最大化（即 $\ln \pi$ 尽可能小，也就是概率分布尽可能扁平）。
    - **博弈：**
        - $Q$ 这一项想让概率集中在最高分的动作上（变尖，低熵）。
        - $\alpha \ln \pi$ 这一项想让概率分散开（变扁，高熵）。
        - **Loss 就是在这两者之间找平衡点。**

### (3) Alpha Loss (自动熵调节)

目的： 自动决定 $\alpha$ 应该多大。如果当前 Agent 太浪了（熵太高），就调大 $\alpha$ 惩罚它；如果太保守（熵太低），就调小 $\alpha$ 鼓励它。

注：实际代码中通常优化的是 $\log \alpha$，为了保证 $\alpha$ 始终为正。

#### 核心公式

$$L(\alpha) = \mathbb{E}_{a \sim \pi} [ -\alpha (\ln \pi(a|s) + \bar{H}) ]$$

#### 逐项拆解

1. **$\bar{H}$ (Target Entropy / 目标熵)**
    
    - **含义：** 我们预设的一个“最低随机性底线”。
        
    - **常用值：** `-dim(Action Space)`（动作维度的相反数）。例如，如果控制机器人的 6 个关节，目标熵就是 -6。
        
    - **直觉：** 我们希望 Agent 的探索程度至少要达到这个标准。
        
2. **$\ln \pi(a|s)$ (当前实际熵的负数)**
    
    - 这是 Agent 当前策略实际表现出来的随机程度。
        
3. **括号内的逻辑 $(\ln \pi + \bar{H})$**
    
    - **情况 A：熵太小（太确定）。**
        
        - $\ln \pi$ 是比如 -2（概率大），$\bar{H}$ 是 -6。
            
        - $(-2) + (-6)$ ... 等等，这里是比较绝对值。
            
        - 直观理解：如果 $\ln \pi$ (实际熵的逆) **大于** 目标，说明熵太低了。这部分差值需要被修正。
            
4. **$-\alpha$ (梯度方向)**
    
    - 通过梯度下降最小化 Loss。
        
    - 如果实际熵 < 目标熵：$\alpha$ 会**变大**（增加探索权重）。
        
    - 如果实际熵 > 目标熵：$\alpha$ 会**变小**（减少探索权重，允许收敛）。

---

### 5. 实现方式与步骤 (Implementation Loop)

SAC 是 **Off-policy** 算法，通常配合 **Replay Buffer (经验回放池)** 使用。

#### 训练流程 (Online Training Loop):

1. **初始化：** Actor, Critics, Target Critics, Replay Buffer。
    
2. **环境交互 (Rollout)：**
    
    - 观察状态 $s$。
        
    - Actor 输出分布，**随机采样**动作 $a$ (保留随机性用于探索)。
        
    - 执行 $a$，获得 $r, s', done$。
        
    - 存入 Replay Buffer。
        
3. **梯度更新 (Update)：**
    
    - 当 Buffer 数据足够时，随机采样一个 Batch。
        
    - **更新 Critic：** 计算 Target Q，计算 MSE Loss，反向传播更新 $Q_1, Q_2$。
        
    - **更新 Actor：** 利用重参数化技巧采样动作，计算 Q 值和 LogProb，反向传播更新 $\pi$。
        
    - **更新 Alpha：** 根据当前熵与目标熵的差异更新 $\alpha$。
        
    - **Soft Update：** 更新 Target Critic 参数 ($\theta_{targ} \leftarrow \tau \theta + (1-\tau)\theta_{targ}$)。
        

#### 预测/测试流程 (Inference):

在模型训练好后进行测试或部署时：

- **不进行采样！** 直接使用高斯分布的**均值 (Mean)** 作为动作。
    
- 这是为了保证表现的确定性和最优性。
    

---

### 6. 具体代码示例 (PyTorch 风格)

为了简洁，只展示核心逻辑部分。

Python

```
import torch
import torch.nn.functional as F

class SAC_Agent:
    def __init__(self, state_dim, action_dim):
        # 1. 初始化网络
        self.actor = Actor(state_dim, action_dim)
        self.q1 = SoftQNetwork(state_dim, action_dim)
        self.q2 = SoftQNetwork(state_dim, action_dim)
        # 目标网络
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        
        # 自动熵调节
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def update(self, batch):
        s, a, r, s_next, d = batch
        
        # --- 1. 更新 Critic ---
        with torch.no_grad():
            # 计算下一个状态的动作和熵
            next_action, log_prob, _ = self.actor.sample(s_next)
            
            # 计算 Target Q
            q1_next = self.q1_target(s_next, next_action)
            q2_next = self.q2_target(s_next, next_action)
            q_min = torch.min(q1_next, q2_next)
            
            # 这里的 alpha 是动态的
            alpha = self.log_alpha.exp()
            
            # 核心公式：y = r + gamma * (Q - alpha * log_pi)
            target_q = r + (1 - d) * self.gamma * (q_min - alpha * log_prob)
        
        # 当前 Q
        current_q1 = self.q1(s, a)
        current_q2 = self.q2(s, a)
        
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Critic 梯度下降...
        
        # --- 2. 更新 Actor ---
        # 重新采样动作 (带梯度)
        new_action, log_prob, _ = self.actor.sample(s)
        q1_new = self.q1(s, new_action)
        q2_new = self.q2(s, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor Loss: 最小化 (alpha * log_prob - Q) -> 最大化 (Q + 熵)
        actor_loss = (alpha * log_prob - q_new).mean()
        
        # Actor 梯度下降...
        
        # --- 3. 更新 Alpha ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        # Alpha 梯度下降...
```

---

### 7. 分类辨析：On-policy, Off-policy, Online, Offline

这是面试和应用中容易混淆的点：

1. **Off-policy (离策略)：**
    
    - **SAC 是 Off-policy。**
        
    - **定义：** 学习用的数据可以由**其他策略**（比如过去的旧策略）生成。
        
    - **体现：** SAC 使用 Replay Buffer，从过去存的历史数据中随机采样进行训练，而不是只使用当前最新策略产生的数据。
        
2. **On-policy (在线策略)：**
    
    - PPO, TRPO 是 On-policy。它们只能利用当前策略产生的样本进行一次更新，之后样本必须丢弃，效率较低。
        
3. **Online RL (在线强化学习)：**
    
    - **SAC 通常是 Online 的。**
        
    - **定义：** Agent 在训练过程中**不断与环境交互**，产生新数据存入 Buffer。
        
    - **流程：** 交互 -> 存Buffer -> 训练 -> 交互...
        
4. **Offline RL (离线强化学习)：**
    
    - **标准 SAC 在纯 Offline 场景下表现不佳。**
        
    - **定义：** 不允许与环境交互，只给定一个静态的 Dataset 进行训练。
        
    - **问题：** SAC 会遭受 OOD (Out-of-Distribution) 问题，即 Actor 会针对数据集中没见过的动作产生高估的 Q 值（幻觉）。
        
    - **变体：** 针对 Offline 场景，有专门改进的算法如 **CQL (Conservative Q-Learning)**，它是在 SAC 基础上加了保守项惩罚。
        

### 8. 总结：SAC 的优缺点

**优点：**

- **样本效率极高：** 相比 PPO，需要的交互步数少很多（通常少 10 倍以上）。
    
- **对超参数不敏感：** 自适应 Alpha 机制让调参变得很简单。
    
- **稳定性好：** 结合了 Off-policy 的效率和最大熵的稳定性。
    

**缺点：**

- **计算量大：** 需要同时更新两个 Q 网络、一个 Actor 和一个 Alpha，且每次推断都要从高斯分布采样。
    
- **主要是连续动作：** 虽然有针对离散动作的改版（SAC-Discrete），但原生设计是针对连续控制的（如机器人手臂控制、自动驾驶）。
    

**适用场景：**

- 机器人控制（MuJoCo, PyBullet）。

- 自动驾驶车辆控制。

- 任何仿真成本较高、需要尽可能少交互就能学会的连续控制任务。

## 分类与相关算法

- **RL分类**:  > [[02-模块/Actor-Critic/]] 
- **核心理论**: 最大熵强化学习
- **数据来源**: [[Online-RL]], (变体用于 [[Offline-RL]])
- **动作空间**: [[Continuous-Actions]]
- **学习范式**: 
- **相关算法**:
  - [[DDPG]] (前身)
  - [[TD3]] (改进)
  - [[CQL]] (离线变体)