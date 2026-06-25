---
tags:
  - module
  - model-free
  - value-based
  - off-policy
  - online-rl
  - discrete
  - tabular
created: 2026-06-25
---

# Q-Learning（Q-学习）

> 最经典的无模型强化学习算法之一。1989年Watkins提出，用"异策略"方式直接学习最优策略的Q函数，无需环境模型，是后续DQN等深度RL算法的理论根基。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/Bellman方程]]：Q函数递归分解的数学基础
- [[01-原子/TD误差]]：用采样回报逼近期望回报的核心思想
- [[01-原子/epsilon-greedy]]：探索与利用的平衡策略
- **Q表（Q-table）**：用表格存储所有(状态, 动作)对的Q值（Q-Learning特有）

---

## 核心创新

### 异策略（Off-Policy）学习最优Q函数

Q-Learning的关键突破：**用当前策略收集数据，但学习的是最优策略的Q函数**。

**比喻："实习生 vs 理想CEO"**

想象一个实习生（行为策略，epsilon-greedy）每天在公司做事，偶尔瞎搞（探索），但他在心里默默计算："如果我是完美CEO（目标策略，greedy），在同样情况下会怎么选？"——他学到的不是"实习生能赚多少"，而是"完美决策能赚多少"。

这就是 Off-Policy 的精髓：**行为策略（behavior policy）和目标策略（target policy）分离**。

- 行为策略：$\epsilon$-greedy（有 $\epsilon$ 概率随机探索）
- 目标策略：$\text{argmax}_{a'} Q(s', a')$（纯greedy，选最大Q值的动作）

### 为什么Q-Learning能收敛到最优

只要每个(状态, 动作)对被无限次访问，且学习率满足 Robbins-Monro 条件：
$$\sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty$$
Q-Learning **保证收敛**到最优Q函数 $Q^*$。

---

## 算法流程

```
初始化：Q表 Q(s, a) = 0（对所有 s, a）

对于每个episode：
    初始化状态 s

    对于每个步骤：
        1. 用 ε-greedy 根据 Q(s, ·) 选择动作 a
           （以 1-ε 概率选 argmax Q(s,a)，以 ε 概率随机选）
        2. 执行 a，观察即时奖励 r 和新状态 s'
        3. 计算 TD 目标：
               y = r + γ · max_{a'} Q(s', a')    ← 注意这里是 max，不是实际选的动作
        4. 更新 Q 表：
               Q(s, a) ← Q(s, a) + α · (y - Q(s, a))
        5. s ← s'
```

### 关键区别（与 [[SARSA]] 对比）

|  | Q-Learning | [[SARSA]] |
|---|---|---|
| 目标值计算 | $r + \gamma \cdot \max_{a'} Q(s', a')$ | $r + \gamma \cdot Q(s', a')$ |
| 策略类型 | Off-Policy（学最优，不管行为） | On-Policy（学当前策略） |
| 风格 | 乐观激进（永远假设下一步最优） | 保守务实（下一步按当前策略走） |

---

## 关键公式

### Q-Learning 更新规则（核心）

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

各项含义：
- $\alpha$：学习率，控制更新步长
- $r$：即时奖励
- $\gamma$：折扣因子，未来奖励的权重
- $\max_{a'} Q(s', a')$：下一状态的最优Q值（贪心取最大）
- 方括号内即 **TD误差**：$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$

### Bellman最优方程（理论基础）

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \cdot \max_{a'} Q^*(s', a') \right]$$

Q-Learning的更新规则正是对这个方程的**随机逼近（stochastic approximation）**。

### 最优策略提取

$$\pi^*(s) = \text{argmax}_{a} Q^*(s, a)$$

---

## 优缺点

### 优点
- 理论保证收敛到最优策略（在表格、充分探索条件下）
- Off-Policy：可以用任意策略收集数据，灵活性高
- 实现简单，概念清晰，是RL入门首选
- 不需要环境模型（model-free）

### 缺点
- 只能处理有限离散状态空间（Q表无法扩展到高维）
- 高维状态空间需要函数近似（→ [[DQN]]）
- 收敛速度慢（表格型，无泛化能力）
- 对超参数（学习率、ε衰减策略）敏感
- 连续动作空间不直接适用

---

## 直觉理解：迷宫老鼠

想象一只老鼠（agent）在迷宫（环境）里找奶酪（奖励）。

**Q表 = 老鼠脑中的"地图"**：每个路口（状态）的每个方向（动作）都有一个评分（Q值）。

**学习过程**：
1. 老鼠从起点出发，在每个路口，大多数时候走评分最高的方向，偶尔随机探索（ε-greedy）
2. 找到奶酪（正奖励）或踩到陷阱（负奖励）后，沿着来路更新每个路口的评分
3. **关键**：更新时，老鼠想的是"下一个路口，**最好的**方向能赚多少"——不管它实际上次走了哪个方向
4. 经过足够多次探索，每个路口的评分都趋于真实价值，老鼠的最短路径自然浮现

**Off-Policy 的体现**：老鼠即使某次在路口A往左走了（随机探索），更新路口A的"往右"评分时，仍然用"下一个路口最优方向的评分"来计算——它学的是"如果我很聪明，该怎么走"，而不是"我这次实际怎么走的"。

---

## 与 SARSA 在悬崖边的经典对比

 Sutton《强化学习》中的经典 cliff-walking 例子：

```
S  .  .  .  .  .  .  .  .  .  .  G
   ─────────────────────────────
          悬崖（掉下去 -100）
```

- **Q-Learning**：学最优策略，贴着悬崖边走（因为这是最短路径），但探索时偶尔掉下悬崖，代价惨重
- **[[SARSA]]**：学当前策略，知道探索时会掉悬崖，主动绕远路走安全路线

**结论**：Q-Learning学出的策略更优，但执行时风险更高；SARSA更保守但更鲁棒。

---

## 演化位置

**在 Value-Based 演化链中的位置**：

```
Bellman方程（1950s，理论基础）
    ↓
Q-Learning（1989，表格型Off-Policy）  ← 你在这里
    ↓
DQN（2015，神经网络近似 + 经验回放 + 目标网络）
    ↓
DDQN（2016，解决高估问题）
    ↓
Dueling DQN / Rainbow DQN
    ↓
CQL（2020，离线RL版本）
```

详见 [[03-流程/Value-Based演化史]]

---

## 代码片段

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        self.epsilon = epsilon  # 探索概率

    def choose_action(self, state):
        # ε-greedy 策略
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        # 当前Q值
        current_q = self.Q[state, action]

        # TD目标：注意是 max（Off-Policy核心）
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        # 更新
        self.Q[state, action] += self.alpha * (target - current_q)

    def get_policy(self):
        # 提取最优策略
        return np.argmax(self.Q, axis=1)
```

---

## 参考资源

- 原始论文：Watkins, C.J.C.H. "Learning from Delayed Rewards" (PhD thesis, 1989)
- 教科书：Sutton & Barto《Reinforcement Learning: An Introduction》第6章
- 收敛性证明：Watkins & Dayan "Q-Learning" (Machine Learning, 1992)

---

## 相关算法

- [[SARSA]]：On-Policy对应算法，学习当前策略而非最优策略
- [[DQN]]：将Q表替换为神经网络，进入深度RL时代
- [[DDQN]]：解决Q-Learning/DQN的高估问题
- [[02-模块/Value-Based/CQL|CQL]]：Q-Learning在离线RL场景的保守版本
