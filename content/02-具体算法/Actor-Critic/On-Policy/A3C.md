---
tags:
  - #model-free
  - #actor-critic
  - #on-policy
  - #online-rl
  - #continuous
  - #function-approximation
---

## [Link] 知识图谱链接

### 相关算法
- [[A2C]]
- [[PPO]]

### 分类维度
- [[01-分类维度/Actor-Critic-Methods]]
- [[On-Policy]]
- [[01-分类维度/Function-Approximation]]


# Asynchronous Advantage Actor-Critic (A3C)

A3C is an actor-critic algorithm that uses asynchronous parallel training.

## Overview
Multiple agents train in parallel on different copies of the environment.

## Key Features
- Asynchronous updates
- Advantage function for reduced variance
- Better exploration and stability

## Related
- [[01-分类维度/Actor-Critic-Methods]]
- [[On-Policy]]
- [[A2C]]
- [[PPO]]