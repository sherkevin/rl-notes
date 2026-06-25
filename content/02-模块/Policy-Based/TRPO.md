---
tags:
  - #model-free
  - #policy-based
  - #on-policy
  - #online-rl
  - #continuous
  - #function-approximation
created: 2026-06-25
---

## [Link] 知识图谱链接

### 相关算法
- [[PPO]]
- [[REINFORCE]]

### 分类维度
- [[02-模块/Policy-Based/PPO|Policy-Based算法]]
- [[00-索引/按策略类型|On-Policy]]

---

# Trust Region Policy Optimization (TRPO)

> 信任域策略优化：通过 KL 散度硬约束限制每次策略更新的幅度，从数学上保证策略的单调改进。PPO 的理论前身。

---

## 组合构成

本模块由以下原子组合而成：
- [[01-原子/策略梯度]]：直接优化策略参数
- [[01-原子/优势函数]]：降低策略梯度方差
- [[01-原子/KL散度]]：衡量新旧策略之间的距离，作为信任域约束
- [[01-原子/Trust-Region]]：限制更新步长，保证局部近似可靠
- [[01-原子/重要性采样]]：允许用旧策略数据估计新策略性能
- **Actor-Critic 架构**：Actor 学习策略，Critic 学习状态值函数

---

## 核心创新

TRPO 解决的核心问题：**策略梯度更新步子太大怎么办？**

普通策略梯度（如 REINFORCE）的一个致命缺陷是——更新步长难以选择。步长太小，学习慢得令人绝望；步长太大，策略性能可能突然崩溃，而且很难恢复（尤其在复杂环境中，一个"变傻了"的策略可能再也采不到好数据）。

**核心思想**：Schulman et al. (2015) 从理论上推导出一个**策略性能改进的下界**。只要新旧策略之间的 KL 散度足够小，就能保证新策略**至少不会比旧策略差**。

用一句话概括：**"每次更新策略时，确保新策略和旧策略足够相似，这样进步就是稳的、单调的。"**

---

## 算法流程

### 1. 采样轨迹

使用当前策略 $\pi_{\theta_\text{old}}$ 与环境交互，收集轨迹数据：

```
对于每个 episode:
    1. 从 π_{θ_old} 采样动作，记录 (s_t, a_t, r_t)
    2. 同时记录 log π_{θ_old}(a_t|s_t) 用于重要性采样
```

### 2. 计算优势估计

使用 Critic 网络 $V_\phi$ 计算 GAE 优势：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

### 3. 构建并求解约束优化问题

**目标函数**（替代优势，Surrogate Advantage）：

$$L(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \hat{A}_t \right]$$

**约束条件**（KL 散度硬约束）：

$$\bar{D}_{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

其中 $\delta$ 是信任域半径（通常取 0.01）。

### 4. 求解方法：共轭梯度 + 线搜索

```
1. 用共轭梯度法（Conjugate Gradient）求解自然梯度方向：
   F · x = g
   其中 F 是 Fisher 信息矩阵，g 是策略梯度
   （不需要显式构建 F，只需能计算 F · v 的矩阵向量积）

2. 计算步长：
   Δθ = √(2δ / (x^T · F · x)) · x

3. 线搜索（Line Search）：
   从 θ_old + Δθ 开始，逐步缩小步长（乘以衰减系数 α）
   直到满足：
   (a) L(θ_new) > L(θ_old)   // 目标函数改善
   (b) D_KL(θ_old || θ_new) ≤ δ  // KL 约束满足
```

### 5. 更新 Critic

用收集的数据更新值函数 $V_\phi$（标准回归问题）。

### 6. 重复

用新策略 $\pi_\theta$ 重新采样数据（On-policy，旧数据丢弃）。

---

## 关键公式

### 策略改进下界（理论核心）

$$\eta(\pi') \geq \eta(\pi) + \sum_s \rho_{\pi'}(s) \sum_a \pi'(a|s) A_\pi(s,a) - C \cdot D_{KL}^{\max}(\pi \| \pi')$$

- $\eta(\pi)$：策略 $\pi$ 的期望回报
- $A_\pi(s,a)$：优势函数
- $C = \frac{4\epsilon\gamma}{(1-\gamma)^2}$，其中 $\epsilon$ 是优势函数的最大绝对值
- **含义**：只要 KL 散度足够小，右边第二项（惩罚项）就很小，策略改进就有保证

### 替代目标函数（Surrogate Objective）

$$L_{\theta_\text{old}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_\text{old}}, a \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} A_{\theta_\text{old}}(s, a) \right]$$

- 重要性采样比 $\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}$ 允许用旧策略的数据来评估新策略
- 这个目标函数在 $\theta = \theta_\text{old}$ 处与真实目标函数一阶等价

### KL 散度约束

$$\mathbb{E}_{s \sim \rho_{\theta_\text{old}}} [D_{KL}(\pi_{\theta_\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s))] \leq \delta$$

- **为什么用 KL 散度？** 因为 KL 散度是策略空间中"自然"的距离度量——它衡量的是"新策略相对于旧策略的惊讶程度"。
- **为什么约束而不是惩罚？** 硬约束能保证**每次更新都不退化**；软惩罚（如 KL penalty 系数）需要精心调参，效果不如硬约束稳健。

### Fisher 信息矩阵与自然梯度

$$F = \mathbb{E}_{s,a} [\nabla_\theta \log \pi_\theta(a|s) \cdot (\nabla_\theta \log \pi_\theta(a|s))^T]$$

自然梯度方向：$\tilde{g} = F^{-1} g$

- **直觉**：普通梯度 $g$ 是"在参数空间中最陡的方向"；自然梯度 $F^{-1}g$ 是"在策略概率分布空间中最陡的方向"。后者更合理，因为参数的微小变化不一定对应策略的微小变化。

---

## 优缺点

### 优点
- **单调改进保证**：理论上保证每次更新策略都不会变差（在近似误差范围内）
- **更新步长自动确定**：不需要手动调学习率，信任域半径 $\delta$ 代替了这个角色
- **稳定性好**：在复杂任务（如 MuJoCo 连续控制）上比 vanilla PG 稳定得多
- **理论基础扎实**：从策略改进下界出发，每一步都有数学支撑

### 缺点
- **实现复杂**：需要实现共轭梯度法、Fisher 矩阵向量积、线搜索等
- **计算开销大**：每步需要多次前向/后向传播（线搜索）
- **样本效率低**：On-policy，每批数据只用一次
- **扩展性受限**：共轭梯度法在大规模参数下效率下降
- **被 PPO 取代**：PPO 用简单的 Clip 机制达到了相似甚至更好的效果，实现容易得多

---

## 演化位置

**在策略梯度演化链中的位置**：

```
REINFORCE (1992)
    ↓
策略梯度定理 (1999)
    ↓
Natural PG (2002)：Fisher 信息矩阵做预条件
    ↓
TRPO (2015)：KL 散度硬约束 ← 你在这里
    ↓
PPO (2017)：Clip 软约束，简化 TRPO
    ↓
GRPO (2024)：去 Critic
```

详见 [[03-流程/Policy-AC演化史]]

---

## 真实例子与直觉理解

**直觉类比**：想象你在调收音机频道。

- **普通策略梯度**：猛地一转旋钮——可能直接跳过了好频道，再也找不回来。
- **TRPO**：小心地微转旋钮，每转一点就听听效果，确保比刚才好听才继续。转的幅度由"和刚才差多远"来决定。
- **PPO**：给旋钮加个物理限位器（Clip），转的时候最多只能转一小格。效果差不多，但简单得多。

**经典实验**：MuJoCo HalfCheetah
- TRPO 通常能稳定学到 ~3000 分
- 但实现难度大，代码量是 PPO 的 3-5 倍
- 在同样的超参数调优预算下，PPO 往往能超过 TRPO

---

## TRPO vs PPO 对比

| 特性 | TRPO | PPO |
|------|------|-----|
| 约束方式 | KL 散度硬约束 | Clip 软约束 |
| 优化方法 | 共轭梯度 + 线搜索 | 标准 SGD/Adam |
| 实现复杂度 | 高 | 低 |
| 理论保证 | 单调改进 | 启发式 |
| 实际效果 | 稳健 | 通常更好（调参后）|
| 计算开销 | 高（矩阵向量积 + 线搜索）| 低 |

---

## 代码片段

```python
# TRPO 核心逻辑（伪代码）
import torch

def trpo_update(policy, old_policy, states, actions, advantages):
    # 1. 计算策略梯度 g
    ratio = torch.exp(policy.log_prob(actions) - old_policy.log_prob(actions))
    surrogate_loss = (ratio * advantages).mean()
    g = torch.autograd.grad(surrogate_loss, policy.parameters())
    g = torch.cat([grad.view(-1) for grad in g])
    
    # 2. 定义 Fisher 矩阵向量积函数
    def fisher_vector_product(v):
        # KL 散度的 Hessian 矩阵乘以向量 v
        kl = kl_divergence(old_policy, policy).mean()
        kl_grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])
        kl_v = (kl_grad * v).sum()
        kl_hessian = torch.autograd.grad(kl_v, policy.parameters())
        return torch.cat([h.view(-1) for h in kl_hessian])
    
    # 3. 共轭梯度法求解 F^{-1} g
    step_dir = conjugate_gradient(fisher_vector_product, g, nsteps=10)
    
    # 4. 计算步长
    max_step = torch.sqrt(2 * delta / (step_dir @ fisher_vector_product(step_dir)))
    full_step = max_step * step_dir
    
    # 5. 线搜索
    for fraction in [1.0, 0.5, 0.25, 0.125]:
        new_params = old_params + fraction * full_step
        update_policy(policy, new_params)
        
        new_ratio = torch.exp(policy.log_prob(actions) - old_policy.log_prob(actions))
        new_loss = (new_ratio * advantages).mean()
        new_kl = kl_divergence(old_policy, policy).mean()
        
        if new_loss > old_loss and new_kl <= delta:
            break  # 找到可接受的步长
    else:
        update_policy(policy, old_params)  # 线搜索失败，回退
```

---

## 分类与相关算法

- **RL分类**：Model-free, Policy-Based
- **数据来源**：On-Policy, Online-RL
- **动作空间**：Discrete / Continuous
- **相关算法**：
  - [[02-模块/Policy-Based/REINFORCE|REINFORCE]]（基础策略梯度，无约束）
  - [[02-模块/Policy-Based/PPO|PPO]]（Clip 简化版 TRPO）
  - [[02-模块/Actor-Critic/A3C|A3C]] / [[02-模块/Actor-Critic/A2C|A2C]]（并行化策略梯度）

---

## 参考资源

- 论文：Schulman et al. (2015). "Trust Region Policy Optimization." *ICML*.
- 论文：Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*.
- 代码：[OpenAI Spinning Up TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
- 代码：[ModularRL TRPO](https://github.com/joschu/modular_rl)（Schulman 原版）
- 博客：[OpenAI Blog - Trust Region Policy Optimization](https://openai.com/research/trpo)
