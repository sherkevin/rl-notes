# Value-Based Methods (基于价值的方法)

## 定义
学习价值函数 Q(s,a) 或 V(s)，通过价值函数间接推导策略。

## 核心思想
评估每个状态-动作对的价值，策略是通过选择价值最高的动作得到的：
```
π(a|s) = argmax_a Q(s, a)
```

## 所有算法
```dataview
LIST
WHERE contains(tags, "#value-based")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 分类

### 按策略类型
- **On-Policy**: [[SARSA]]
- **Off-Policy**: [[Q-Learning]], [[DQN]], [[DDQN]], [[CQL]]

### 按学习范式
- **Tabular**: [[Q-Learning]], [[SARSA]]
- **函数逼近**: [[DQN]], [[DDQN]], [[CQL]]

## 优缺点

### 优点
- ✅ 价值估计相对稳定
- ✅ 适合离散动作空间
- ✅ 理论基础扎实

### 缺点
- ❌ 只能处理离散动作
- ❌ 无法学习随机策略

## 相关分类
- [[01-分类维度/Policy-Based-Methods]]
- [[01-分类维度/Actor-Critic-Methods]]
- [[01-分类维度/按优化对象分类]]

## 参见
- [[📋 Value-Based算法汇总]]
