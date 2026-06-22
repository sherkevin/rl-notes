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
- [[01-分类维度/Policy-Based-Methods]]
- [[On-Policy]]
- [[01-分类维度/Function-Approximation]]


# REINFORCE

REINFORCE is a Monte Carlo policy gradient algorithm for policy-based RL.

## Overview
Updates policy parameters by computing gradients from complete episode trajectories.

## Formula
∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) * R_t]

## Related
- [[01-分类维度/Policy-Based-Methods]]
- [[On-Policy]]
- [[PPO]]
- [[TRPO]]