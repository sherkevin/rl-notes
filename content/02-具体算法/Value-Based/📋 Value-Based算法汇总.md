# Value-Based 算法汇总

> 基于价值的强化学习方法

## 定义
学习价值函数 Q(s,a)，通过选择价值最高的动作来决策。

## 所有算法

```dataview
TABLE
  策略类型 as "策略类型",
  数据来源 as "数据来源",
  动作空间 as "动作空间",
  范式 as "范式"
FROM "02-具体算法"
WHERE contains(tags, "#value-based")
SORT file.name ASC
```

## 按策略分类

### On-Policy
```dataview
LIST
WHERE contains(tags, "#value-based")
AND contains(tags, "#on-policy")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

### Off-Policy
```dataview
LIST
WHERE contains(tags, "#value-based")
AND contains(tags, "#off-policy")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 核心概念
- [[Bellman方程和TD误差]]
- [[经验回放]]
- [[目标网络]]

## 相关分类
- [[01-分类维度/按优化对象分类]]
- [[Policy-Based算法汇总]]
- [[Actor-Critic算法汇总]]
