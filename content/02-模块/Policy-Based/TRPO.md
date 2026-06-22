---
tags:
  - #model-free
  - #policy-based
  - #on-policy
  - #online-rl
  - #continuous
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[PPO]]
- [[REINFORCE]]

### 分类维度
- [[01-分类维度/Policy-Based-Methods]]
- [[On-Policy]]
- [[01-分类维度/Function-Approximation]]


# Trust Region Policy Optimization (TRPO)

TRPO is a policy optimization algorithm that constrains policy updates within a trust region.

## Overview
Uses KL divergence constraint to ensure policy updates are conservative and stable.

## Related
- [[01-分类维度/Policy-Based-Methods]]
- [[On-Policy]]
- [[PPO]]
- [[REINFORCE]]
- [[00-基础概念/Forward-KL与Reverse-KL]] — TRPO 显式约束 Reverse KL 以保证单调改进