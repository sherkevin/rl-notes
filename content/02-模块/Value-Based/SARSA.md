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
- [[01-分类维度/Value-Based-Methods]]
- [[On-Policy]]
- [[00-基础概念/Tabular-Methods]]


# SARSA

SARSA is an on-policy temporal difference learning algorithm for value-based RL.

## Overview
Updates Q-values using the action actually taken by the current policy.

## Formula
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

## Related
- [[01-分类维度/Value-Based-Methods]]
- [[On-Policy]]
- [[Q-Learning]]
- [[DQN]]