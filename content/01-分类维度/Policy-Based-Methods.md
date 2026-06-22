# Policy-Based Methods (基于策略的方法)

## 定义
直接参数化策略 π(a|s,θ)，通过优化策略参数来最大化期望回报。

## 核心思想
直接输出动作概率分布，使用策略梯度方法更新：
```
θ ← θ + α·∇J(θ)
```

## 所有算法
```dataview
LIST
WHERE contains(tags, "#policy-based")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 分类

### 按策略类型
- **On-Policy**: [[REINFORCE]], [[PPO]], [[TRPO]]
- **Off-Policy**: [[02-具体算法/Policy-Based/Policy-Gradient]], [[TD3+BC]], [[AWAC]]

### 按动作空间
- **离散**: [[REINFORCE]]
- **连续**: [[PPO]], [[TRPO]]

## 优缺点

### 优点
- ✅ 可以处理连续动作空间
- ✅ 可以学习随机策略
- ✅ 策略收敛可能更快
- ✅ 适合高维动作空间

### 缺点
- ❌ 训练不稳定，方差大
- ❌ 容易陷入局部最优
- ❌ 样本效率低

## 相关分类
- [[01-分类维度/Value-Based-Methods]]
- [[01-分类维度/Actor-Critic-Methods]]
- [[01-分类维度/按优化对象分类]]

## 参见
- [[📋 Policy-Based算法汇总]]
