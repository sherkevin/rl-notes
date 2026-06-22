---
tags:
  - #model-free
  - #value-based
  - #off-policy
  - #online-rl
  - #discrete
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[DQN]]
- [[Q-Learning]]

### 分类维度
- [[02-模块/Value-Based/]]
- [[Off-Policy]]
- 


# Double DQN (DDQN)

Double DQN is an improvement over standard DQN that addresses overestimation bias.

## Overview
Uses separate networks for action selection and evaluation to reduce maximization bias.

## Key Innovation
Decouples action selection from value evaluation using target and main networks.

## Related
- [[DQN]]
- [[02-模块/Value-Based/]]
- [[Off-Policy]]