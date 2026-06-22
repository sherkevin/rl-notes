# Actor-Critic Methods (演员-评论家方法)

## 定义
结合了Value-Based和Policy-Based的优点，同时学习策略（Actor）和价值函数（Critic）。

## 核心思想

### Actor (演员)
- 负责选择动作
- 学习策略 π(a|s)
- 输出动作分布

### Critic (评论家)
- 负责评估动作价值
- 学习 Q(s,a) 或 V(s)
- 指导Actor更新

### 协作机制
```
Actor 选择动作 → 环境执行 → Critic 评估
       ↑                           ↓
       └─────── 根据评估更新 ───────┘
```

## 所有算法
```dataview
LIST
WHERE contains(tags, "#actor-critic")
AND file.path =~ "02-具体算法/"
SORT file.name ASC
```

## 分类

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

## 优缺点

### 优点
- ✅ 结合了两者优点
- ✅ 比纯策略方法更稳定
- ✅ 比纯价值方法更高效
- ✅ 适合连续动作空间

### 缺点
- ❌ 实现更复杂
- ❌ 需要同时优化两个网络
- ❌ 超参数敏感

## 相关分类
- [[01-分类维度/Value-Based-Methods]]
- [[01-分类维度/Policy-Based-Methods]]
- [[01-分类维度/按优化对象分类]]

## 参见
- [[📋 Actor-Critic算法汇总]]
