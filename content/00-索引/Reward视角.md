---
tags:
  - index
  - reward-perspective
created: 2026-06-22
---

# Reward 视角

> 从奖励建模的角度，串联 RL 算法的演化脉络。关注"如何定义和学习好的奖励"。

---

## 核心问题

**如何定义和学习一个奖励信号，引导智能体学习期望行为？**

这个核心问题衍生出多个子问题：
1. 如何从人类偏好学习奖励？→ [[01-原子/Reward-Model训练方法|Reward Model]]
2. 如何从结果学习奖励？→ ORM
3. 如何从过程学习奖励？→ PRM
4. 如何避免训练 RM？→ [[02-模块/Reward-Model/DPO|DPO]], [[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]]
5. 如何让模型自我评估？→ [[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]], [[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]]

---

## 演化脉络

### 第一阶段：人工设计奖励 (1950s-2017)

```
人工设计奖励函数
    ↓
Reward Shaping (1999)：添加辅助奖励
    ↓
Curiosity-driven Exploration (2017)：内在动机
```

**核心问题**：奖励函数难以设计，需要领域专家

---

### 第二阶段：从人类偏好学习 (2017-2022)

```
RLHF (2017)：人类反馈强化学习
    ↓
Pairwise RM (2022)：Bradley-Terry模型，成对比较
    ↓
Pointwise RM (2023)：直接打分
```

**关键文档**：
- [[01-原子/Bradley-Terry模型]]：偏好建模的理论基础
- [[02-模块/Reward-Model/Pairwise-RM]]：InstructGPT使用的方法
- [[02-模块/Reward-Model/Pointwise-RM]]：直接打分

**核心问题**：标注成本高，只能评估最终结果

---

### 第三阶段：过程监督 (2023-2024)

```
ORM (2023)：只看最终结果
    ↓
PRM (2024)：逐步评分
    ↓
Math-Shepherd (2024)：自动生成PRM数据
```

**关键文档**：
- [[02-模块/Reward-Model/ORM]]：Outcome [[01-原子/Reward-Model训练方法|Reward Model]]
- [[02-模块/Reward-Model/PRM]]：Process [[01-原子/Reward-Model训练方法|Reward Model]]
- [[02-模块/Reward-Model/Self-Rewarding]]：模型自我评估

**核心问题**：PRM标注成本更高，需要自动化

---

### 第四阶段：[[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] (2024)

```
LLM-as-Judge (2024)：用大模型评估
    ↓
Generative RM (2024)：生成式奖励模型
```

**关键文档**：
- [[02-模块/Reward-Model/LLM-as-Judge]]：GPT-4/Claude评估
- [[02-模块/Reward-Model/Generative-RM]]：生成理由+分数

**核心问题**：LLM评估成本高，需要更高效的方案

---

### 第五阶段：隐式奖励 (2023-2025)

```
DPO (2023)：策略即奖励，隐式RM
    ↓
Online DPO (2024)：在线版本
    ↓
SimPO (2024)：无需参考模型
```

**关键文档**：
- [[02-模块/Reward-Model/DPO]]：隐式RM的开创工作
- [[01-原子/KL散度]]：DPO的理论基础

**核心问题**：隐式RM无法动态调整奖励

---

### 第六阶段：可验证奖励 (2024-2025)

```
Verifiable Reward (2024)：规则验证
    ↓
DeepSeek-R1 (2024)：数学/代码验证
    ↓
Search-R1 (2025)：检索准确率验证
```

**关键文档**：
- [[02-模块/Reward-Model/Verifiable-Reward]]：规则验证
- [[02-模块/Policy-Based/GRPO]]：使用verifiable reward

**核心问题**：只适用于有确定性答案的任务

---

## 关键技术对比

| 技术 | 解决的问题 | 代表算法 |
|------|-----------|---------|
| **[[01-原子/Bradley-Terry模型|Bradley-Terry]]** | 偏好建模 | [[02-模块/Reward-Model/Pairwise-RM|Pairwise RM]] |
| **过程监督** | 中间步骤奖励 | PRM |
| **结果监督** | 最终结果奖励 | ORM |
| **LLM评估** | 自动化评估 | [[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] |
| **生成式RM** | 可解释性 | [[02-模块/Reward-Model/Generative-RM|Generative RM]] |
| **隐式RM** | 避免训练RM | [[02-模块/Reward-Model/DPO|DPO]] |
| **可验证奖励** | 无需人工 | [[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]] |
| **自我评估** | 减少依赖 | [[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]] |

---

## 奖励建模方法对比

| 方法 | 标注成本 | 训练成本 | 适用范围 | 可解释性 |
|------|---------|---------|---------|---------|
| [[02-模块/Reward-Model/Pairwise-RM|Pairwise RM]] | 高 | 中 | 通用 | 低 |
| [[02-模块/Reward-Model/Pointwise-RM|Pointwise RM]] | 高 | 中 | 通用 | 低 |
| PRM | 极高 | 中 | 推理任务 | 中 |
| ORM | 低 | 中 | 推理任务 | 低 |
| [[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] | 低 | 无 | 通用 | 高 |
| [[02-模块/Reward-Model/Generative-RM|Generative RM]] | 中 | 中 | 通用 | 高 |
| [[02-模块/Reward-Model/DPO|DPO]] | 中 | 低 | 偏好数据 | 低 |
| Verifiable | 无 | 无 | 确定性任务 | 高 |
| [[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]] | 低 | 低 | 通用 | 中 |

---

## 学习路线建议

### 入门路线（理解基础）

```
Bradley-Terry模型 → Pairwise RM → Pointwise RM → ORM
```

**预计时间**：2天

**关键收获**：
- 理解偏好建模的理论基础
- 掌握RM的基本训练方法
- 了解ORM的实现

---

### 进阶路线（掌握PRM）

```
ORM → PRM → Math-Shepherd → Self-Rewarding
```

**预计时间**：3-4天

**关键收获**：
- 理解过程监督的优势
- 掌握PRM的数据生成方法
- 学会自动化评估技术

---

### 前沿路线（LLM对齐）

```
Pairwise RM → DPO → LLM-as-Judge → Verifiable Reward → GRPO
```

**预计时间**：5-7天

**关键收获**：
- 理解隐式RM的思想
- 掌握LLM评估方法
- 学会最新的对齐技术

---

## 相关视角

- [[00-索引/Policy视角]]：从策略优化的角度看奖励学习
- [[00-索引/按数据来源]]：Online vs Offline vs Preference Data
- [[00-索引/按模型依赖]]：Model-Free vs Model-Based

---

## 延伸阅读

- [[03-流程/Offline-RL与RLHF演化史]]：完整的演化历史
- [[01-原子/KL散度]]：DPO的理论基础
- [[03-流程/Offline-RL与RLHF演化史|RLHF完整流程]]：SFT → RM → [[02-模块/Policy-Based/PPO|PPO]]/[[02-模块/Reward-Model/DPO|DPO]] 的完整流程

---

## 应用场景指南

### 通用对话/写作

**推荐**：[[02-模块/Reward-Model/Pairwise-RM|Pairwise RM]] + [[02-模块/Policy-Based/PPO|PPO]]

**理由**：
- 人类偏好数据质量高
- 适用于开放式任务
- InstructGPT验证的方法

**参考**：[[02-模块/Reward-Model/Pairwise-RM]]

---

### 数学/推理

**推荐**：[[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]] + [[02-模块/Policy-Based/GRPO|GRPO]]

**理由**：
- 答案可验证，无需RM
- 训练简单高效
- DeepSeek-R1验证的方法

**参考**：[[02-模块/Reward-Model/Verifiable-Reward]], [[02-模块/Policy-Based/GRPO]]

---

### 代码生成

**推荐**：[[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]] (test cases) + PRM

**理由**：
- 代码可执行验证
- PRM提供中间步骤奖励
- 适合长序列任务

**参考**：[[02-模块/Reward-Model/PRM]]

---

### 快速原型

**推荐**：[[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] → 训小RM → [[02-模块/Policy-Based/PPO|PPO]]

**理由**：
- 标注成本低（GPT-4生成）
- 快速验证想法
- 可以迭代改进

**参考**：[[02-模块/Reward-Model/LLM-as-Judge]]

---

### 资源有限

**推荐**：[[02-模块/Reward-Model/DPO|DPO]]

**理由**：
- 无需训练RM
- 无需PPO的复杂流程
- 训练简单

**参考**：[[02-模块/Reward-Model/DPO]]

---

### 需要可解释性

**推荐**：[[02-模块/Reward-Model/Generative-RM|Generative RM]]

**理由**：
- 生成评估理由
- 可调试和改进
- 适合高要求场景

**参考**：[[02-模块/Reward-Model/Generative-RM]]

---

### 数据不足

**推荐**：[[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]]

**理由**：
- 自我生成数据
- 迭代改进
- 减少人工依赖

**参考**：[[02-模块/Reward-Model/Self-Rewarding]]
