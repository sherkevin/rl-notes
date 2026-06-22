---
tags:
  - foundational
  - reward-model
  - rlhf
  - preference-learning
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

这个分数被 RL 算法（PPO、GRPO 等）用来引导策略优化。Reward model 的质量直接决定了 RLHF 的上限——**garbage reward in, garbage policy out**。

---

## 方法一：Pairwise 训练（Bradley-Terry 模型）

这是 InstructGPT (2022) 的做法，也是目前最主流的方式。

### 数据格式

人工标注员看到一个 prompt 和**两个** response，选一个更好的：

```
Prompt: "解释什么是黑洞"
Response A: "黑洞是时空中的一个区域..."  ← 标注员选这个
Response B: "黑洞就是一个很黑很黑的洞..."
```

### 训练目标（Bradley-Terry 模型）

假设人类选择 A 的概率是：

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中 $\sigma$ 是 sigmoid 函数，$y_w$ 是赢的 response，$y_l$ 是输的。

Loss 函数：

$$L(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**直觉**：让赢的 response 得分尽量高于输的，差距越大越好。

### 优缺点

- ✅ 人类标注简单直观（二选一比打分容易）
- ✅ Bradley-Terry 模型理论成熟
- ❌ 只学到**序**（ranking），没学到**绝对值**——两个 response 差 0.1 分和差 10 分，模型不知道
- ❌ 标注成本高，每个偏好对都需要人工比较

---

## 方法二：Pointwise 训练 —— 直接打分

### 数据格式

每个 (prompt, response) 对直接给一个标量分数：

```
Prompt: "解释什么是黑洞"
Response: "黑洞是时空中的一个区域..."
Score: 7.5 / 10
```

### 训练目标

直接回归：

$$L(\phi) = \mathbb{E}_{(x, y, s)} \left[ (r_\phi(x, y) - s)^2 \right]$$

### 优缺点

- ✅ 能学到绝对分数（知道"好"和"非常好"的差距）
- ❌ 人类打分**极其不一致**——不同标注员对同一个 response 可能给 3 分和 8 分
- ❌ 标注员很难校准自己的标准

### 实际怎么用

实践中很少纯用 pointwise，通常是 **pairwise 为主 + pointwise 为辅**：先用 pairwise 训 ranking，再用少量 pointwise 数据校准绝对值。

---

## 方法三：Process Reward Model (PRM) vs Outcome Reward Model (ORM)

这是 2024 年以来最重要的区分，尤其在推理/数学任务上。

### ORM（Outcome Reward Model）—— 只看最终结果

```
Prompt: "证明根号2是无理数"
Response: [10 行推理过程] → 结论: "所以根号2是无理数"
Reward: 1.0 (正确) 或 0.0 (错误)
```

- ✅ 标注简单（对/错二分类）
- ❌ **credit assignment 问题**：10 步推理，哪一步做对了？哪一步做错了？ORM 不知道
- ❌ 奖励稀疏：错了就是 0，没有中间信号

### PRM（Process Reward Model）—— 给每一步打分

Let's Verify Step by Step (Lightman 2023, OpenAI) 提出：

```
Prompt: "证明根号2是无理数"
Step 1: "假设根号2是有理数，即 √2 = a/b"  → Reward: 0.9 (好的开始)
Step 2: "那么 2 = a²/b²"                     → Reward: 0.8 (正确推导)
Step 3: "所以 a² = 2b²"                      → Reward: 0.85 (对)
Step 4: "所以 a 和 b 都是偶数"               → Reward: 0.3 (跳步了，需要更多论证)
```

训练数据需要标注员给**每一步**打分（或标对错）。

- ✅ 密集的奖励信号，每步都有反馈
- ✅ 在数学推理上显著优于 ORM（Let's Verify 论文：PRM 通过率 78% vs ORM 56%）
- ❌ 标注成本极高（每步都要标）
- ❌ 如何定义"一步"在不同任务中不统一

### PRM 的训练

PRM 通常也训练为 pointwise 分类器——对每一步预测"这一步到最终答案正确的概率"：

$$L(\phi) = -\mathbb{E} \sum_t \left[ y_t \log r_\phi(x, y_{\leq t}) + (1-y_t) \log(1 - r_\phi(x, y_{\leq t})) \right]$$

其中 $y_t \in \{0, 1\}$ 是第 t 步的标注标签。

### 2024-2025 发展

- **Math-Shepherd** (2024)：用 MCTS 自动生成每步的 reward 标注，不需要人工标 PRM 数据
- **OmegaPRM** (2024)：改进 PRM 数据生成流程
- **PRM 的 reward 可以来自验证器**：数学题直接 check 每步是否合法，代码题跑 test case

---

## 方法四：LLM-as-Judge —— 不训 RM，直接用大模型打分

### 思路

用一个强 LLM（GPT-4、Claude）直接给 response 打分或做偏好判断，**不训练任何模型**。

```
System Prompt: "你是一个评估专家。请对以下回答打分 (1-10):"
User: "Prompt: ... Response: ..."
Judge: "Score: 7/10, reasoning: 回答准确但缺少例子..."
```

### 变体

| 变体 | 方式 |
|---|---|
| **Pointwise Judge** | 直接给单个 response 打分 |
| **Pairwise Judge** | 比较两个 response，选一个 |
| **Reference-guided** | 提供参考答案，让 judge 对照评分 |

### 优缺点

- ✅ 零训练成本，开箱即用
- ✅ 能给出理由（可解释性）
- ❌ **推理成本极高**——每次评分都是一次 LLM 推理
- ❌ 有位置偏好（pairwise 中倾向选第一个/最后一个）
- ❌ 有冗长偏好（倾向给更长的 response 高分）
- ❌ 不可微分，不能直接用在 RL 训练中（通常用来生成训练数据，再训一个小 RM）

### 实际怎么用

LLM-as-Judge 主要用于**生成偏好数据**：用 GPT-4 批量对比 response 对，生成偏好数据，再用这些数据训一个小 RM。这比人工标注便宜得多。

---

## 方法五：Generative Reward Model —— 先生成理由再打分

### 思路

传统 RM 是一个黑盒标量输出。Generative RM 让模型**先输出评价理由，再输出分数**：

```
Input: "Prompt: ... Response: ..."
Output: "这个回答的优点是准确、有条理。缺点是缺少具体例子。
         综合评分: 7/10"
```

### 代表工作

- **Skywork-Reward** (2024)：generative RM + RLHF
- **Self-Taught Evaluators** (Meta 2024)：让模型自我训练评判能力
- **Generative Reward Models** (2024, arxiv 2410.12832)：证明 generative RM 在某些场景下优于 Bradley-Terry RM

### 训练

两阶段：
1. 用 LLM 生成 (prompt, response, critique, score) 的训练数据
2. 微调一个较小模型学习"生成 critique + 输出分数"

### 优缺点

- ✅ 可解释——能看到模型为什么给这个分
- ✅ critique 本身可以作为过程奖励
- ❌ 推理速度慢（要生成一段文本而不只是一个数）

---

## 方法六：Verifiable Reward —— 规则验证，不需要训练

对于有**确定性答案**的任务，直接用规则验证：

| 任务 | 验证方式 |
|---|---|
| 数学 | 检查最终答案是否正确（exact match 或 sympy 验证）|
| 代码 | 跑 test cases，看 pass rate |
| 格式化输出 | 正则匹配格式是否正确 |
| 事实性 | 对照知识库 check |

### 在 RL 中的使用

这就是 DeepSeek-R1 和 GRPO 系方法用的奖励：

```python
def reward_fn(prompt, response):
    answer = extract_answer(response)
    ground_truth = get_ground_truth(prompt)
    return 1.0 if answer == ground_truth else 0.0
```

- ✅ 完全不需要训练 reward model
- ✅ 奖励信号完美（对就是对，错就是错）
- ❌ 只适用于有确定性答案的任务
- ❌ 无法评估开放式回答（写作、对话、创意）

### 2025-2026 趋势

越来越多的工作用 **verifiable reward + RL** 来训练推理能力，绕过 reward model 训练：
- DeepSeek-R1：数学/代码用 verifiable reward
- OpenAI o1/o3：推理链 + verifiable reward
- **Search-R1** (2025)：用检索准确率作为 verifiable reward

---

## 方法七：Self-Rewarding Model —— 模型自己给自己打分

### 思路

让 LLM 自己评估自己的输出质量：

```
Prompt: "评价以下回答的质量 (1-5 分):"
Response: "..."
Self-score: "4/5, because..."
```

### 代表工作

- **Self-Rewarding Language Models** (Meta 2024)：模型同时作为策略和 reward model，迭代自我改进
- **Constitutional AI** (Anthropic)：模型根据一组原则自我评估

### 训练

迭代过程：
1. 模型生成多个 response
2. 模型自己评估哪个更好（LLM-as-Judge on itself）
3. 用这些偏好数据微调模型
4. 回到步骤 1，用更强的模型生成更好的偏好数据

### 优缺点

- ✅ 减少对人工标注的依赖
- ✅ 可以无限迭代（每轮模型变强 → 偏好数据质量更高）
- ❌ 容易"自嗨"——模型倾向给自己生成的内容高分
- ❌ 需要外部校准防止 reward hacking

---

## 方法八：不训 RM —— DPO 系（隐式 Reward Model）

DPO (2023) 的数学发现：**最优 RLHF 策略有 closed-form 解**，不需要显式训练 reward model。

$$L_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**直觉**：DPO 把"训练 reward model + 用 RM 做 RL"这两步合并成一步直接优化偏好。模型本身"暗含"了一个 reward model：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

### 优缺点

- ✅ 不需要训练 RM，不需要 PPO，不需要 RL 循环
- ✅ 训练极其简单（就是一个二分类 loss）
- ❌ 只用离线偏好数据，不做在线探索
- ❌ 不能动态调整奖励信号

详见 [[04-演化综述/Offline-RL与RLHF演化史]]。

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

### 场景选择

| 场景 | 推荐方法 |
|---|---|
| 通用对话/写作对齐 | Pairwise BT RM + PPO |
| 数学/推理 | Verifiable reward + GRPO（不需要 RM）|
| 代码生成 | Verifiable reward（跑 test）+ PRM（逐步 debug）|
| 快速原型 | LLM-as-Judge 生成数据 → 训小 RM |
| 资源有限 | DPO（跳过 RM 训练）|
| 需要可解释性 | Generative RM（先给理由再打分）|
| 数据不足 | Self-Rewarding（迭代自我改进）|

### 2025-2026 趋势

1. **Verifiable reward 在推理领域碾压 trained RM**——DeepSeek-R1、o1/o3 证明了纯规则奖励 + RL 的效果
2. **PRM 越来越重要**——过程监督比结果监督对复杂推理更有效
3. **Generative RM 替代黑盒 RM**——可解释性成为刚需
4. **LLM-as-Judge 成为数据生产主力**——用 GPT-4/Claude 批量生成偏好数据，再训小 RM
5. **DPO 系 + 在线探索**（Online DPO、SimPO）在简单对齐任务上够用

---

## 相关文档

- [[00-基础概念/Forward-KL与Reverse-KL]]：RL 中 KL 约束的理论基础
- [[02-具体算法/Policy-Based/PPO]]：PPO 如何使用 reward model 的分数
- [[02-具体算法/Policy-Based/GRPO]]：GRPO 如何去掉 reward model（Critic）
- [[04-演化综述/Offline-RL与RLHF演化史]]：从 RLHF 到 DPO 到 GRPO 的完整演化
