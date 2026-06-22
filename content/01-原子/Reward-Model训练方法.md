---
tags:
  - index
  - reward-model
  - rlhf
aliases:
  - Reward Model
  - 奖励模型
created: 2026-06-22
---

# Reward Model 训练方法

> **一句话**：Reward model 将人类偏好转化为可计算的标量分数，引导 RL 策略优化。训练方法按反馈粒度和建模方式分为八类，从经典 pairwise 到 2025 年的 verifiable reward。

---

## 核心任务

Reward model 要解决的问题：**给定一个 prompt 和一个 response，输出一个标量分数，表示这个 response 有多"好"。**

$$r_\phi(x, y) \to \mathbb{R}$$

这个分数被 RL 算法（[[02-模块/Policy-Based/PPO|PPO]]、[[02-模块/Policy-Based/GRPO|GRPO]] 等）用来引导策略优化。Reward model 的质量直接决定了 RLHF 的上限——**garbage reward in, garbage policy out**。

---

## 八种训练方法

| 方法 | 数据来源 | 核心思路 | 详见 |
|---|---|---|---|
| **[[02-模块/Reward-Model/Pairwise-RM|Pairwise RM]]** | 人工偏好对 | [[01-原子/Bradley-Terry模型|Bradley-Terry]] 模型，成对比较 | [[02-模块/Reward-Model/Pairwise-RM]] |
| **[[02-模块/Reward-Model/Pointwise-RM|Pointwise RM]]** | 人工打分 | 直接回归标量分数 | [[02-模块/Reward-Model/Pointwise-RM]] |
| **PRM** | 逐步标注 | 给推理过程的每一步打分 | [[02-模块/Reward-Model/PRM]] |
| **ORM** | 结果标注 | 只看最终结果打分 | [[02-模块/Reward-Model/ORM]] |
| **[[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]]** | 大模型代理 | 直接用强 LLM 打分 | [[02-模块/Reward-Model/LLM-as-Judge]] |
| **[[02-模块/Reward-Model/Generative-RM|Generative RM]]** | 生成式评估 | 先输出理由，再打分 | [[02-模块/Reward-Model/Generative-RM]] |
| **[[02-模块/Reward-Model/Verifiable-Reward|Verifiable Reward]]** | 规则验证 | 自动化检查答案正确性 | [[02-模块/Reward-Model/Verifiable-Reward]] |
| **[[02-模块/Reward-Model/DPO|DPO]]** | 偏好数据 | 隐式 reward，不训 RM | [[02-模块/Reward-Model/DPO]] |
| **[[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]]** | 自我评估 | 模型自己给自己打分 | [[02-模块/Reward-Model/Self-Rewarding]] |

---

## 全景总结

```
                    Reward Model 训练方法
                           │
         ┌─────────┬───────┼───────┬─────────┬──────────┐
         │         │       │       │         │          │
    人工偏好数据   人工打分  逐步标注  大模型代理   规则验证    不训 RM
         │         │       │       │         │          │
    Pairwise   Pointwise   PRM   LLM-as    Verifiable   DPO
    (Bradley-   (MSE      (逐步    Judge    (数学/代码   (隐式 RM)
     Terry)    回归)     分类)   + Generative  直接 check)
         │                    │        RM
         │                    │
    InstructGPT          Let's Verify
    RLHF-PPO             Math-Shepherd
                         OpenAI o1
```

---

## 场景选择

| 场景 | 推荐方法 |
|---|---|
| 通用对话/写作对齐 | Pairwise BT RM + [[02-模块/Policy-Based/PPO|PPO]] |
| 数学/推理 | Verifiable reward + [[02-模块/Policy-Based/GRPO|GRPO]]（不需要 RM）|
| 代码生成 | Verifiable reward（跑 test）+ PRM（逐步 debug）|
| 快速原型 | [[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] 生成数据 → 训小 RM |
| 资源有限 | [[02-模块/Reward-Model/DPO|DPO]]（跳过 RM 训练）|
| 需要可解释性 | [[02-模块/Reward-Model/Generative-RM|Generative RM]]（先给理由再打分）|
| 数据不足 | [[02-模块/Reward-Model/Self-Rewarding|Self-Rewarding]]（迭代自我改进）|

---

## 2025-2026 趋势

1. **Verifiable reward 在推理领域碾压 trained RM**——DeepSeek-R1、o1/o3 证明了纯规则奖励 + RL 的效果
2. **PRM 越来越重要**——过程监督比结果监督对复杂推理更有效
3. **[[02-模块/Reward-Model/Generative-RM|Generative RM]] 替代黑盒 RM**——可解释性成为刚需
4. **[[02-模块/Reward-Model/LLM-as-Judge|LLM-as-Judge]] 成为数据生产主力**——用 GPT-4/Claude 批量生成偏好数据，再训小 RM
5. **[[02-模块/Reward-Model/DPO|DPO]] 系 + 在线探索**（Online [[02-模块/Reward-Model/DPO|DPO]]、SimPO）在简单对齐任务上够用

---

## 相关文档

- [[01-原子/KL散度]]：RL 中 KL 约束的理论基础
- [[01-原子/Bradley-Terry模型]]：偏好建模的理论基础
- [[02-模块/Policy-Based/PPO]]：[[02-模块/Policy-Based/PPO|PPO]] 如何使用 reward model 的分数
- [[02-模块/Policy-Based/GRPO]]：[[02-模块/Policy-Based/GRPO|GRPO]] 如何去掉 reward model（Critic）
- [[03-流程/Offline-RL与RLHF演化史]]：从 RLHF 到 [[02-模块/Reward-Model/DPO|DPO]] 到 [[02-模块/Policy-Based/GRPO|GRPO]] 的完整演化
