---
tags:
  - #model-free
  - #value-based
  - #on-policy
  - #online-rl
  - #discrete
  - #tabular
---

## [Link] 知识图谱链接

### 相关算法
- [[Q-Learning]]
- [[PPO]]

### 分类维度
- [[02-模块/Value-Based/]]
- [[On-Policy]]
- 


# SARSA

SARSA is an on-policy temporal difference learning algorithm for value-based RL.

## Overview
Updates Q-values using the action actually taken by the current policy.

## Formula
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

## Related
- [[02-模块/Value-Based/]]
- [[On-Policy]]
- [[Q-Learning]]
- [[DQN]]