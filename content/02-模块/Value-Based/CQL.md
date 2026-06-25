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
- [[02-模块/Value-Based/DQN|Value-Based算法]]
- [[00-索引/按策略类型|Off-Policy]]
- [[00-索引/按数据来源|Offline-RL]]
- 


# CQL（Conservative Q-Learning）

CQL is an offline RL algorithm that addresses out-of-distribution issues in Q-learning.

## Overview
Adds a conservative penalty to prevent overestimation of unseen actions in offline datasets.

## Related
- [[00-索引/按数据来源|Offline-RL]]
- [[Q-Learning]]
- [[SAC]]
- [[02-模块/Value-Based/DQN|Value-Based算法]]
- 完整演化 ([[02-模块/Value-Based/BCQ|BCQ]]→CQL→[[02-模块/Value-Based/IQL|IQL]]→DT): 见 [Offline-RL与RLHF-偏好优化演化全景.md](../../Offline-RL与RLHF-偏好优化演化全景.md#a3-cql-conservative-q-learning-2020)