# Policy-Based 算法汇总

> 基于策略的强化学习方法

## 定义
直接参数化策略 π(a|s)，通过优化策略参数来最大化期望回报。

## 所有算法

```dataview
TABLE
  策略类型 as "策略类型",
  数据来源 as "数据来源",
  动作空间 as "动作空间",
  范式 as "范式"
FROM "02-具体算法"
WHERE contains(tags, "#policy-based")
SORT file.name ASC
```

## 算法列表
```dataview
LIST
WHERE contains(tags, "#policy-based")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 核心特点
- ✅ 可以处理连续动作空间
- ✅ 可以学习随机策略
- ✅ 策略收敛可能更快
- ❌ 训练方差大，不稳定

## 相关分类
- [[01-分类维度/按优化对象分类]]
- [[Value-Based算法汇总]]
- [[Actor-Critic算法汇总]]
