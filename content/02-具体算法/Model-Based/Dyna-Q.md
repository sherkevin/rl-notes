---
tags:
  - #model-based
  - #value-based
  - #off-policy
  - #online-rl
  - #discrete
  - #tabular
---

## [Link] 知识图谱链接

### 相关算法
- [[Q-Learning]]

### 分类维度
- [[Model-Based]]
- [[00-基础概念/Tabular-Methods]]


# Dyna-Q

Dyna-Q is a model-based reinforcement learning algorithm that combines model learning with direct RL.

## Overview
Dyna-Q learns a model of the environment from experience and uses it to generate additional simulated experiences for learning.

## Key Features
- Combines model-based planning with model-free learning
- Uses a learned model to simulate transitions
- Improves sample efficiency

## Related
- [[01-分类维度/Model-Based]]
- [[Q-Learning]]
- [[MBPO]]
- [[PETS]]