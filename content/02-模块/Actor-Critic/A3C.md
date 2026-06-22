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
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|On-Policy]]
- 


# Asynchronous Advantage Actor-Critic (A3C)

A3C is an actor-critic algorithm that uses asynchronous parallel training.

## Overview
Multiple agents train in parallel on different copies of the environment.

## Key Features
- Asynchronous updates
- Advantage function for reduced variance
- Better exploration and stability

## Related
- [[02-模块/Actor-Critic/SAC|Actor-Critic算法]]
- [[00-索引/按策略类型|On-Policy]]
- [[A2C]]
- [[PPO]]