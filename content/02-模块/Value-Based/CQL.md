---
tags:
  - module
  - model-free
  - value-based
  - off-policy
  - offline-rl
  - discrete
  - function-approximation
created: 2026-06-25
---

# CQL（Conservative Q-Learning，保守Q学习）

> 离线强化学习的里程碑算法。2020年Kumar等人提出，通过在Q值上添加保守惩罚项，系统性地压低分布外（OOD）动作的Q值，让策略不敢"自作聪明"选数据集中没见过的动作。一举解决了Offline RL中Q值高估导致策略崩溃的核心难题。

---

## 组合构成

本模块由以下原子组合而成：
- [[02-模块/Value-Based/DQN]]：Q值函数近似框架
- [[01-原子/经验回放]]：离线数据集本质上是固定的经验回放池
- [[02-模块/Actor-Critic/SAC]]：CQL常与SAC结合（连续动作空间版本）
- **保守正则化**：惩罚OOD动作的高Q值（CQL特有）

---

## 核心创新

### 离线RL的核心难题：分布外动作的Q值高估

在线RL（如DQN）：策略选的每个动作，下一步都能与环境交互得到真实反馈。
离线RL：数据集是固定的（比如人类演示、历史日志），策略只能从这批数据学。

**问题**：如果策略想选一个在数据集中很少见或没见过的动作（OOD动作），Q网络对它的估计**不可靠**——往往因为函数近似的外推误差而给出虚高的Q值。策略就会被这些"虚假高分"诱导，选出实际效果很差的动作。

**比喻："只看菜单点菜"**

想象你只看过一份餐厅的菜单（离线数据集），菜单上有番茄炒蛋（常见）和一些你没见过的菜（OOD）。如果你对没见过的菜凭空想象，觉得"没吃过的一定更好吃"（Q值高估），你就可能点出一道难吃的菜。

CQL的策略：**没吃过的菜，默认给它打个折扣**（保守惩罚），让你倾向于点菜单上确实好吃的菜。

### CQL的解法：Q值的保守正则化

在标准Q-Learning的损失函数上，加一个正则化项：

**最小化 OOD 动作的Q值，同时最大化数据集中动作的Q值。**

数学上：让策略 $\pi$ 对状态 $s$ 的期望Q值，减去数据集中实际动作的Q值，这个差越小越好。

---

## 算法流程

```
输入：离线数据集 D = {(s, a, r, s')}（固定，不与环境交互）

初始化：Q网络 Q(θ)，策略网络 π(φ)

对于每个训练步骤：
    从 D 采样一批数据 {(s, a, r, s')}

    1. 计算标准TD目标（与DQN/SAC相同）：
           y = r + γ · Q(s', π(s'); θ⁻)

    2. 计算CQL保守惩罚项：
           对每个状态 s，采样一组候选动作 {a_i}（从当前策略π和均匀分布各采一半）
           计算：log_sum_exp Q(s, a_i; θ) - Q(s, a_data; θ)
           （第一项≈对OOD动作Q值的上界估计，第二项是数据集中动作的Q值）

    3. 总损失 = 标准Q损失 + α · CQL惩罚项
           L(θ) = E[(y - Q(s,a;θ))²] + α · E[log∑exp Q(s,a_i) - Q(s,a_data)]

    4. 更新策略网络 π（最大化保守Q值）：
           L(φ) = -E[Q(s, π(s); θ)]
```

---

## 关键公式

### CQL 核心目标（简化版）

$$\min_\theta \; \underbrace{\alpha \cdot \mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_{a} \exp Q_\theta(s, a) - \mathbb{E}_{a \sim \mathcal{D}}[Q_\theta(s, a)]\right]}_{\text{保守惩罚项（CQL特有）}} + \underbrace{\frac{1}{2}\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - \hat{\mathcal{B}}^\pi Q_\theta(s,a))^2\right]}_{\text{标准Bellman误差}}$$

各项含义：
- $\log \sum_a \exp Q_\theta(s, a)$：对所有候选动作Q值的LogSumExp（近似 $\max$），惩罚高Q值的动作
- $\mathbb{E}_{a \sim \mathcal{D}}[Q_\theta(s, a)]$：数据集中实际动作的Q值期望，鼓励它保持准确
- $\alpha$：保守系数，控制惩罚强度（太大→过于保守，太小→抑制不足）
- $\hat{\mathcal{B}}^\pi$：Bellman算子的经验估计

### CQL 的直觉理解

保守惩罚项的效果：
- 对数据集中常见的动作：Q值由TD目标锚定，相对准确
- 对OOD动作（数据集中少见）：Q值被惩罚压低，策略不敢选

**结果**：$\forall s, \; \mathbb{E}_{a \sim \pi}[Q(s,a)] \leq V^\pi(s)$（策略期望的Q值是真实价值的**下界**）

这就是"保守"的含义：**宁可低估，不冒高估的风险**。

### 策略提取

$$\pi = \text{argmax}_\pi \; \mathbb{E}_{s \sim \mathcal{D}}[Q_\theta(s, \pi(s))]$$

因为Q值是保守的（下界），最大化这个下界相当于在"最坏估计"下找最好的策略。

---

## 优缺点

### 优点
- 系统性解决离线RL的Q值高估问题（理论保证Q值下界）
- 不需要行为克隆（BC）预训练，纯Q-Learning框架
- 在Atari、连续控制等基准上大幅超越BC和其他Offline RL方法
- 可与SAC结合处理连续动作空间（CQL-SAC）
- 概念清晰，实现相对简单（在SAC基础上加一个正则项）

### 缺点
- 保守系数 $\alpha$ 需要仔细调参（太大过于保守，太小不够保守）
- LogSumExp的近似计算需要对动作采样（连续动作空间下有计算开销）
- 对数据质量敏感：如果离线数据本身很差，保守策略也学不到好策略
- 在数据覆盖较好的场景下，过度保守反而限制了性能上限
- 不如[[02-模块/Value-Based/IQL|IQL]]简洁（IQL完全避开OOD动作的Q值估计）

---

## 直觉理解：为什么离线RL这么难

**在线RL（DQN）**：策略选了动作A → 环境给出真实反馈 → Q网络学到A的真实价值

**离线RL（没有CQL）**：策略想选动作B（数据集中没见过）→ Q网络凭空估计B的价值 → 函数近似外推误差 → B被高估为100分（实际只有30分）→ 策略崩溃

**离线RL + CQL**：策略想选动作B（数据集中没见过）→ CQL对B施加保守惩罚 → B的Q值被压低到合理范围 → 策略转向数据集中确实表现好的动作A → 策略稳健

**类比：医疗诊断**

一个AI只看过1000个感冒病例（离线数据），没见过某种罕见病。
- **无CQL**：AI看到一组不典型症状，可能"幻觉"这是某种它能治的病，给出高风险处方
- **有CQL**：AI对没见过的症状组合自动打折扣，倾向于推荐自己确实见过的、安全的治疗方案

---

## 与 BCQ 和 IQL 的对比

| | BCQ（2018） | CQL（2020） | IQL（2021） |
|---|---|---|---|
| 核心思想 | 限制策略在数据集动作附近 | 压低OOD动作Q值 | 完全不查OOD动作的Q值 |
| OOD处理方式 | 生成模型约束动作空间 | 正则化惩罚 | Expectile回归（只看数据内动作） |
| 实现复杂度 | 中（需要VAE生成模型） | 低（加一个正则项） | 低（换损失函数） |
| 保守程度 | 中 | 可调（α控制） | 不保守（但避免OOD） |

---

## 代码片段

```python
import torch
import torch.nn.functional as F

def cql_loss(q_net, states, actions, rewards, next_states, dones, policy_net,
             target_q_net, gamma=0.99, alpha=1.0, num_random=10):
    """
    CQL损失计算（简化版，离散动作空间）
    """
    # 标准TD损失
    with torch.no_grad():
        next_q = target_q_net(next_states).max(dim=1)[0]
        target_q = rewards + gamma * next_q * (1 - dones.float())

    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    td_loss = F.mse_loss(current_q, target_q)

    # CQL保守惩罚
    # 第一项：对所有动作的Q值LogSumExp（惩罚高Q值）
    q_all_actions = q_net(states)  # shape: [batch, n_actions]
    logsumexp_q = torch.logsumexp(q_all_actions, dim=1)  # log Σ exp(Q(s,a))

    # 第二项：数据集中动作的Q值（鼓励保持准确）
    q_data = current_q  # Q(s, a_data)

    # 保守惩罚 = LogSumExp - 数据集Q值
    cql_penalty = (logsumexp_q - q_data).mean()

    # 总损失
    total_loss = td_loss + alpha * cql_penalty
    return total_loss, td_loss.item(), cql_penalty.item()


# 连续动作空间版本（CQL-SAC）需要：
# 1. 从策略π和均匀分布各采样num_random个动作
# 2. 用这些动作的Q值计算LogSumExp近似
```

---

## 演化位置

**在 Offline-RL 演化链中的位置**：

```
在线DQN/SAC（需要环境交互）
    ↓
BCQ（2018，限制动作空间）
    ↓
CQL（2020，保守Q值正则化）  ← 你在这里
    ↓
IQL（2021，Expectile回归，完全避开OOD）
    ↓
Decision Transformer（2021，序列建模范式）
```

详见 [[03-流程/Offline-RL与RLHF演化史]]

---

## 参考资源

- 原始论文：Kumar, Kumar, Fu, Tucker, Levine "Conservative Q-Learning for Offline Reinforcement Learning" (NeurIPS 2020)
- 扩展论文：Kumar et al. "Offline Reinforcement Learning with Implicit Q-Learning" (IQL, 2021)
- 相关：Fujimoto et al. "Off-Policy Deep RL without Exploration" (BCQ, 2018)

---

## 相关算法

- [[02-模块/Value-Based/BCQ|BCQ]]：CQL之前的Offline RL方法，用生成模型限制动作空间
- [[02-模块/Value-Based/IQL|IQL]]：CQL之后的改进，用Expectile回归完全避开OOD问题
- [[Q-Learning]]：CQL的理论基础（保守版Q-Learning）
- [[02-模块/Actor-Critic/SAC|SAC]]：CQL-SAC连续动作版本的基座算法
- [[DQN]]：CQL离散动作版本的基座算法
