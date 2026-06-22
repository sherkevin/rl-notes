---
tags:
  - #model-free
  - #policy-based
  - #on-policy
  - #online-rl
  - #discrete
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[PPO]]
- [[TRPO]]

### 分类维度
- [[02-模块/Policy-Based/PPO|Policy-Based算法]]
- [[00-索引/按策略类型|On-Policy]]
- 


# REINFORCE

REINFORCE is a Monte Carlo policy gradient algorithm for policy-based RL.

## Overview
Updates policy parameters by computing gradients from complete episode trajectories.

## Formula
∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) * R_t]

## Related
- [[02-模块/Policy-Based/PPO|Policy-Based算法]]
- [[00-索引/按策略类型|On-Policy]]
- [[PPO]]
- [[TRPO]]