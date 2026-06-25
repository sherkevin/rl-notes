---
tags:
  - module
  - model-free
  - value-based
  - on-policy
  - online-rl
  - discrete
  - tabular
created: 2026-06-25
---

# SARSA（State-Action-Reward-State-Action）

> 与Q-Learning并列为TD学习的两大经典算法。名字来自它采样的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。On-Policy设计让它学到的策略天然考虑了探索噪声，在风险敏感场景下更鲁棒。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/Bellman方程]]：价值函数递归分解
- [[01-原子/TD误差]]：用采样回报逼近期望
- [[01-原子/epsilon-greedy]]：探索策略，SARSA的行为策略和目标策略相同
- **Q表（Q-table）**：与Q-Learning相同的状态-动作价值表

---

## 核心创新

### On-Policy：学你实际做的事，而不是理想中最好的事

SARSA的关键特征：**行为策略和目标策略是同一个策略**（都是 $\epsilon$-greedy）。

更新时用的下一步动作 $a'$，是当前策略**实际会选**的动作，而不是理论最优动作。

**比喻："真实的实习生 vs 理想CEO"**

与Q-Learning的"实习生学完美CEO决策"不同，SARSA是"实习生学实习生的决策"——它知道自己有 $\epsilon$ 的概率犯错，所以在评估时把这个风险也纳入计算。

**核心区别一句话**：
- Q-Learning更新：$r + \gamma \cdot \max_{a'} Q(s', a')$（假设下一步最优）
- SARSA更新：$r + \gamma \cdot Q(s', a')$（下一步按当前策略走，$a'$ 是实际选的动作）

---

## 算法流程

```
初始化：Q表 Q(s, a) = 0（对所有 s, a）

对于每个episode：
    初始化状态 s
    用 ε-greedy 根据 Q(s, ·) 选择动作 a

    对于每个步骤：
        1. 执行 a，观察奖励 r 和新状态 s'
        2. 用 ε-greedy 根据 Q(s', ·) 选择下一步动作 a'   ← On-Policy关键步骤
        3. 计算 TD 目标：
               y = r + γ · Q(s', a')    ← 用实际选的 a'，不是 max
        4. 更新 Q 表：
               Q(s, a) ← Q(s, a) + α · (y - Q(s, a))
        5. s ← s',  a ← a'
```

### 与 Q-Learning 的伪代码对比

```
# Q-Learning（Off-Policy）
a = ε_greedy(Q, s)
r, s' = env.step(a)
Q(s,a) += α[r + γ · max_a' Q(s', a') - Q(s,a)]   # max！

# SARSA（On-Policy）
a = ε_greedy(Q, s)
r, s' = env.step(a)
a' = ε_greedy(Q, s')                                # 先选下一步动作
Q(s,a) += α[r + γ · Q(s', a') - Q(s,a)]           # 用实际选的 a'！
```

---

## 关键公式

### SARSA 更新规则（核心）

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

其中：
- $a'$ 是当前策略 $\pi$ 在 $s'$ 下**实际选择**的动作（$\epsilon$-greedy）
- 不是 $\max_{a'} Q(s', a')$，而是 $Q(s', a')$

### 期望SARSA（Expected SARSA）

用期望替代采样，减少方差：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \sum_{a'} \pi(a'|s') \cdot Q(s', a') - Q(s, a) \right]$$

期望SARSA是SARSA和Q-Learning的统一框架：
- 当 $\pi$ 是 $\epsilon$-greedy → 期望SARSA
- 当 $\pi$ 是 greedy → 退化为 Q-Learning
- 当 $\pi$ 采样一个动作 → 退化为 SARSA

### Bellman期望方程（理论基础）

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[ r + \gamma \cdot Q^\pi(s', a') \right]$$

SARSA收敛到的是**当前策略 $\pi$ 的Q函数**，而非最优Q函数。

---

## 优缺点

### 优点
- On-Policy：学到的策略考虑了探索风险，在实际执行时更安全
- 在有风险的环境中（如悬崖边）表现更稳健
- 理论收敛保证（在On-Policy条件下）
- 实现简单，与Q-Learning同样易于理解

### 缺点
- 只能学到当前策略的Q值，不直接学最优策略
- 策略改进依赖 $\epsilon$ 逐渐衰减到0（才能逼近最优策略）
- 样本效率相对较低（On-Policy，数据不能跨策略重用）
- 只能处理有限离散状态空间（同Q-Learning）
- 探索不足时可能陷入次优策略

---

## 直觉理解：悬崖行走

Sutton《强化学习》中的经典例子：

```
起点 S  .  .  .  .  .  .  .  .  .  .  终点 G
        ─────────────────────────────
                   悬崖（掉下去 -100）
```

**Q-Learning 的行为**（学最优）：
- 学出的策略：紧贴悬崖边走（最短路径）
- 问题：探索时偶尔掉下悬崖，-100！
- 学出的最优路径收益高，但执行风险大

**SARSA 的行为**（学实际）：
- 它知道自己有 $\epsilon$ 概率随机乱走
- 如果贴悬崖边走，随机乱走就可能掉下去
- 所以它学出的策略：**主动绕远路，远离悬崖边**
- 路径不是最短，但更安全，平均收益反而更高（因为少掉悬崖）

**核心洞察**：
- 当探索成本高（掉悬崖）时，SARSA的保守策略更好
- 当探索成本低、最终要部署最优策略时，Q-Learning更好
- **选择哪个算法，取决于"探索的风险有多大"**

---

## 应用场景

### 适合SARSA的场景
- 探索代价高的环境（机器人控制，摔倒代价大）
- 需要稳健性的在线学习场景
- 安全风险敏感的应用（医疗、自动驾驶）

### 更适合Q-Learning的场景
- 探索成本低（游戏、模拟器）
- 最终要部署最优策略
- 可以用大量模拟数据"烧"探索成本

---

## 代码片段

```python
import numpy as np

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        注意：与Q-Learning不同，需要传入 next_action（On-Policy选择的动作）
        """
        current_q = self.Q[state, action]

        if done:
            target = reward
        else:
            # On-Policy：用实际选的 next_action，不是 max
            target = reward + self.gamma * self.Q[next_state, next_action]

        self.Q[state, action] += self.alpha * (target - current_q)


# 使用示例（与Q-Learning的调用方式对比）
# Q-Learning:
#   agent.update(s, a, r, s', done)          # 内部用 max
# SARSA:
#   a_next = agent.choose_action(s')         # 先选下一步动作
#   agent.update(s, a, r, s', a_next, done)  # 传入实际选的动作
```

---

## 演化位置

**在 Value-Based / On-Policy 演化链中的位置**：

```
Bellman方程（理论基础）
    ↓
SARSA（1994，On-Policy TD学习）  ← 你在这里
    ↓
Expected SARSA（减少方差）
    ↓
SARSA(λ)（引入资格迹，加速信用分配）
    ↓
On-Policy深度RL：[[02-模块/Policy-Based/PPO|PPO]]（策略梯度方向）
```

详见 [[03-流程/Value-Based演化史]]

---

## 参考资源

- 原始论文：Rummery & Niranjan "On-Line Q-Learning Using Connectionist Systems" (1994)
- 教科书：Sutton & Barto《Reinforcement Learning: An Introduction》第6章（悬崖行走例子）
- 收敛性分析：Singh, Jaakkola, Littman, Szepesvári "Convergence Results for Single-Step On-Policy Reinforcement Learning Algorithms" (2000)

---

## 相关算法

- [[Q-Learning]]：Off-Policy对应算法，学习最优策略
- [[DQN]]：将表格Q-Learning扩展到深度RL
- [[02-模块/Policy-Based/PPO|PPO]]：现代On-Policy深度RL代表（策略梯度方向）
