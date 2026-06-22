---
tags:
  - #model-free
  - #value-based
  - #off-policy
  - #offline-rl
  - #discrete
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[DQN]]
- [[SAC]]

### 分类维度
- [[01-分类维度/Value-Based-Methods]]
- [[Off-Policy]]
- [[Offline-RL]]
- [[01-分类维度/Function-Approximation]]


# Conservative Q-Learning (CQL)

CQL is an offline RL algorithm that addresses out-of-distribution issues in Q-learning.

## Overview
Adds a conservative penalty to prevent overestimation of unseen actions in offline datasets.

## Related
- [[Offline-RL]]
- [[Q-Learning]]
- [[SAC]]
- [[01-分类维度/Value-Based-Methods]]
- 完整演化 (BCQ→CQL→IQL→DT): 见 [Offline-RL与RLHF-偏好优化演化全景.md](../../Offline-RL与RLHF-偏好优化演化全景.md#a3-cql-conservative-q-learning-2020)