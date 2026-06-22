# Actor-Critic 算法汇总

> 演员-评论家方法：结合价值与策略

## 定义
同时学习策略（Actor）和价值函数（Critic），Critic的评估用来指导Actor的更新。

## 核心思想
```
Actor (演员)  → 负责选择动作，学习策略 π(a|s)
Critic (评论家) → 负责评估动作价值，学习 Q(s,a)
```

## 所有算法

```dataview
TABLE
  策略类型 as "策略类型",
  数据来源 as "数据来源",
  动作空间 as "动作空间",
  范式 as "范式"
FROM "02-具体算法"
WHERE contains(tags, "#actor-critic")
SORT file.name ASC
```

## 按策略分类

### On-Policy
```dataview
LIST
WHERE contains(tags, "#actor-critic")
AND contains(tags, "#on-policy")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

### Off-Policy
```dataview
LIST
WHERE contains(tags, "#actor-critic")
AND contains(tags, "#off-policy")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 优势
- ✅ 结合了Value-Based和Policy-Based的优点
- ✅ 比纯策略方法更稳定
- ✅ 比纯价值方法更高效

## 相关分类
- [[01-分类维度/按优化对象分类]]
- [[Value-Based算法汇总]]
- [[Policy-Based算法汇总]]
