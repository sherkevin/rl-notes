---
tags:
  - #model-free
  - #value-based
  - #off-policy
  - #online-rl
  - #discrete
  - #tabular
---

## [Link] 知识图谱链接

### 相关算法
- [[DQN]]
- [[SARSA]]
- [[DDQN]]

### 分类维度
- [[02-模块/Value-Based/DQN|Value-Based算法]]
- [[00-索引/按策略类型|Off-Policy]]
- 


# Q-Learning

Q-Learning is a model-free, off-policy temporal difference learning algorithm.

## Overview
Learns the optimal Q-function by updating towards the maximum possible future reward.

## Formula
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

## Related
- [[02-模块/Value-Based/DQN|Value-Based算法]]
- [[00-索引/按策略类型|Off-Policy]]
- [[SARSA]]
- [[DQN]]