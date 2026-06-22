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
- [[01-分类维度/Value-Based-Methods]]
- [[Off-Policy]]
- [[00-基础概念/Tabular-Methods]]


# Q-Learning

Q-Learning is a model-free, off-policy temporal difference learning algorithm.

## Overview
Learns the optimal Q-function by updating towards the maximum possible future reward.

## Formula
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

## Related
- [[01-分类维度/Value-Based-Methods]]
- [[Off-Policy]]
- [[SARSA]]
- [[DQN]]