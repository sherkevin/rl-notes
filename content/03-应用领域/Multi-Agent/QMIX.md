---
tags:
  - #multi-agent
  - #value-based
  - #off-policy
---

# QMIX

> QMIX (Rashid 2018). 单调混合网络进行值函数分解，保证分散执行时与集中决策一致。

## 来源与动机

VDN（Value Decomposition Networks）首次提出值分解框架，将全局 Q 值定义为各智能体 Q 值的简单求和。但 VDN 的线性分解限制了表达能力——只能表示线性值分解，无法捕捉智能体之间的非线性交互。QMIX 引入单调混合网络，在保持 IGM（Individual-Global-Max）性质的前提下大幅提升表达能力。

**论文**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018, Oxford

## 核心创新

用一个**单调性约束的混合网络**将各智能体的 Q 值非线性地组合为全局 Q 值。混合网络的所有权重由超网络 (hypernetwork) 从全局状态 $s$ 生成，并经过非负约束（softplus 或 abs）确保单调性。单调性约束保证了 $\arg\max$ 操作可以分散化——全局最优联合动作等于每个智能体各自最优动作的组合。

## 关键公式与架构

$$Q_{\text{tot}}(s, a) = f_{\text{mix}}(Q_1(o_1, a_1), Q_2(o_2, a_2), \ldots, Q_n(o_n, a_n); s)$$

其中 $f_{\text{mix}}$ 满足单调性约束：
$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0, \quad \forall i$$

**IGM 保证**：因为单调性，$\arg\max_{a_1,\ldots,a_n} Q_{\text{tot}} = (\arg\max_{a_1} Q_1, \ldots, \arg\max_{a_n} Q_n)$，确保集中训练的结果可以分散执行。

## 优缺点

- ✅ 比 VDN 表达力强得多，可以利用全局状态调节分解方式
- ✅ 在 SMAC 上大幅超越 VDN，成为 SMAC benchmark 的标准方法
- ✅ 保证 IGM，梯度直接传递到各智能体
- ❌ 单调性约束仍然限制了表达能力——无法表示某些需要非单调分解的值函数
- ❌ Rashid et al. (2020) 的 Weighted QMIX 后续分析了这个投影偏差局限

## 演化位置

IQL (1993) → VDN (2017) → **QMIX** → Weighted QMIX / QPLEX / QATT
QMIX 是值分解方法的里程碑，成为后续大量工作的基线和比较对象。

## 相关算法

- [[04-演化综述/MARL全景综述|VDN]] — 线性求和分解的前身
- [[MADDPG]] — 策略梯度路线的 CTDE 方法
- [[MAPPO]] — on-policy 路线的强基线
- 完整演化: 见 [MARL全景综述](../../04-演化综述/MARL全景综述.md#b3-qmix-monotonic-value-function-factorisation)
