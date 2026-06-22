# Model-Based 算法汇总

> 基于模型的强化学习方法

## 定义
学习或使用环境动力学模型，利用模型进行规划和策略优化。

## 所有算法

```dataview
TABLE
  策略类型 as "策略类型",
  数据来源 as "数据来源",
  动作空间 as "动作空间",
  范式 as "范式"
FROM "02-具体算法"
WHERE contains(tags, "#model-based")
SORT file.name ASC
```

## 算法列表
```dataview
LIST
WHERE contains(tags, "#model-based")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 核心特点
- ✅ 样本效率高
- ✅ 可以进行规划
- ❌ 模型误差会影响策略
- ❌ 实现复杂

## 相关分类
- [[01-分类维度/按模型依赖]]
- [[Model-Free算法汇总]]
