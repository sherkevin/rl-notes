---
tags:
  - llm
  - mid-training
  - reasoning-rl
  - self-play
  - rl-application
created: 2026-06-26
---

# Mid-Training 阶段：能力学习

> Mid-Training 是 LLM 训练中最具创新性的阶段，目标是在 Base Model 基础上学习特定能力（推理、代码、数学）。这个阶段 RL 是核心驱动力，特别是推理 RL 的突破（如 DeepSeek-R1）。

---

## 阶段目标

**输入**：Base Model（有知识但无特定能力）

**输出**：Capable Model（具备特定能力）
- ✅ 数学推理能力
- ✅ 代码生成能力
- ✅ 逻辑规划能力
- ✅ 长链条推理能力
- ❌ 但仍可能输出不当内容（需要 Post-Training 对齐）

---

## 为什么需要 Mid-Training？

**问题 1：Pre-Training 学到的是知识，不是能力**

```
Pre-Training 学到的：
  - "勾股定理是 a² + b² = c²"（知识）
  - "Python 的 for 循环语法"（知识）

Mid-Training 要学的：
  - 如何用勾股定理解决实际问题（能力）
  - 如何用 Python 实现复杂算法（能力）
```

**问题 2：推理数据稀缺**

- 高质量推理数据需要专家标注，成本极高
- 数学题的解题过程、代码的调试过程，人工标注不可扩展
- 需要模型自己生成训练数据

**问题 3：Post-Training 不适合学复杂能力**

- Post-Training 主要学对齐（有用、无害）
- 推理能力需要大量练习，不是几句对话能学会的
- 需要在 Mid-Training 阶段集中训练

---

## RL 核心方法

### 1. 自我对弈 (Self-Play)

**问题**：高质量推理数据稀缺，人工标注不可扩展。

**核心思想**：让模型自己生成训练数据，形成"自我改进循环"。

#### 基本流程

```
循环：
  1. 模型生成多个候选回答
  2. 用验证器判断正确性
  3. 正确的作为正样本，错误的作为负样本
  4. 用 DPO/PPO 更新模型
  5. 用更新后的模型重复步骤 1
```

#### 数学形式

**Step 1: 生成候选**

对每个 prompt $x$，用当前模型 $\pi_\theta$ 生成 $N$ 个候选回答：

$$\{y_1, y_2, \ldots, y_N\} \sim \pi_\theta(\cdot | x)$$

**Step 2: 验证**

用验证器 $V(x, y)$ 判断每个回答的正确性：

$$V(x, y_i) = \begin{cases} 1 & \text{if } y_i \text{ is correct} \\ 0 & \text{otherwise} \end{cases}$$

**Step 3: 构造偏好对**

从正确回答中选 $y_w$（winner），从错误回答中选 $y_l$（loser）：

$$y_w \in \{y_i : V(x, y_i) = 1\}, \quad y_l \in \{y_i : V(x, y_i) = 0\}$$

**Step 4: DPO 更新**

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**Step 5: 更新参考模型**

$$\pi_{\text{ref}} \leftarrow \pi_\theta$$

#### 代表工作

**SPIN (Self-Play Fine-Tuning)**

**核心创新**：用 DPO 框架实现自我对弈，无需奖励模型。

```python
# SPIN 训练循环
for iteration in range(num_iterations):
    # 1. 用当前模型生成数据
    prompts = load_prompts()
    responses = []
    for prompt in prompts:
        y1 = model.generate(prompt, temperature=1.0)
        y2 = model.generate(prompt, temperature=1.0)
        responses.append((prompt, y1, y2))
    
    # 2. 用验证器判断
    preferences = []
    for prompt, y1, y2 in responses:
        v1 = verifier.verify(prompt, y1)
        v2 = verifier.verify(prompt, y2)
        if v1 and not v2:
            preferences.append((prompt, y1, y2))  # y1 wins
        elif v2 and not v1:
            preferences.append((prompt, y2, y1))  # y2 wins
    
    # 3. DPO 训练
    model.dpo_train(preferences, ref_model=prev_model)
    
    # 4. 更新参考模型
    prev_model = model.copy()
```

**效果**：
- 在 GSM8K（数学推理）上，从 60% 提升到 75%
- 无需人工标注偏好数据

📖 **详细文档**：[[05-应用领域/Mid-Training/SPIN]]

**STaR (Self-Taught Reasoner)**

**核心创新**：用思维链 (Chain-of-Thought) 进行自我对弈。

```
流程：
  1. 给模型数学题，让它生成推理过程
  2. 如果答案正确，保留推理过程作为正样本
  3. 如果答案错误，丢弃
  4. 用正确样本微调模型
  5. 重复
```

**效果**：
- 在 GSM8K 上，从 55% 提升到 70%
- 证明了"推理能力可以通过自我练习提升"

---

### 2. 推理 RL (Reasoning RL)

**问题**：传统 RLHF 只奖励最终答案，不奖励推理过程。但推理能力需要"过程正确"，不仅仅是"答案正确"。

**核心思想**：用[[02-模块/Reward-Model/PRM|过程奖励模型]]（[[02-模块/Reward-Model/PRM|PRM]]）奖励推理的每一步，而不仅仅是最终答案。

#### 传统 RLHF vs 推理 RL

**传统 RLHF**：

```
问题：123 × 456 = ?

模型输出：
  "123 × 456 = 56088"

奖励：
  如果答案正确 → +1
  如果答案错误 → -1

问题：模型可能猜对答案，但推理过程是错的
```

**推理 RL**：

```
问题：123 × 456 = ?

模型输出（带推理过程）：
  "首先计算 123 × 6 = 738"     → 奖励 +0.2
  "然后计算 123 × 50 = 6150"   → 奖励 +0.2
  "最后计算 123 × 400 = 49200" → 奖励 +0.2
  "总和：738 + 6150 + 49200 = 56088" → 奖励 +0.4
  
总奖励：+1.0

问题：如果某一步错了，会立即得到负奖励，而不是等到最后
```

#### 过程奖励模型 (Process Reward Model, PRM)

**训练数据**：

```
问题：求解方程 2x + 5 = 13

推理步骤 1："将 5 移到等式右边" → 正确 → 标签 1
推理步骤 2："2x = 13 - 5 = 8" → 正确 → 标签 1
推理步骤 3："x = 8 / 2 = 4" → 正确 → 标签 1

问题：求解方程 3x - 7 = 11

推理步骤 1："将 -7 移到等式右边" → 正确 → 标签 1
推理步骤 2："3x = 11 + 7 = 18" → 正确 → 标签 1
推理步骤 3："x = 18 / 3 = 5" → 错误 → 标签 0（应该是 6）
```

**PRM 结构**：

```python
class ProcessRewardModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained("bert-base")
        self.classifier = nn.Linear(768, 1)
    
    def forward(self, question, reasoning_step):
        # 拼接问题和推理步骤
        input_text = f"Question: {question}\nStep: {reasoning_step}"
        
        # 编码
        embeddings = self.model(input_text).last_hidden_state[:, 0, :]
        
        # 分类：这一步是否正确
        score = self.classifier(embeddings)
        return sigmoid(score)  # 返回 [0, 1] 概率
```

**训练 PRM**：

```python
# 损失函数：二元交叉熵
def prm_loss(predicted_score, label):
    return -label * log(predicted_score) - (1 - label) * log(1 - predicted_score)

# 训练循环
for question, steps, labels in training_data:
    for step, label in zip(steps, labels):
        score = prm(question, step)
        loss = prm_loss(score, label)
        loss.backward()
```

#### 推理 RL 训练流程

**Step 1: 生成推理轨迹**

```python
def generate_reasoning_trace(model, question, max_steps=10):
    trace = []
    current_context = question
    
    for step_idx in range(max_steps):
        # 生成下一步推理
        next_step = model.generate(
            current_context,
            prompt="Next reasoning step:",
            temperature=0.7
        )
        trace.append(next_step)
        current_context += "\n" + next_step
        
        # 如果生成了最终答案，停止
        if "Final answer:" in next_step:
            break
    
    return trace
```

**Step 2: 用 PRM 评分**

```python
def score_trace(prm, question, trace):
    step_scores = []
    for step in trace:
        score = prm(question, step)
        step_scores.append(score)
    return step_scores
```

**Step 3: 计算优势**

```python
def compute_advantage(step_scores):
    # 方法 1：累积奖励
    cumulative_rewards = []
    for i in range(len(step_scores)):
        cum_reward = sum(step_scores[:i+1])
        cumulative_rewards.append(cum_reward)
    
    # 方法 2：用 GAE
    advantages = gae(step_scores, gamma=0.99, lambda_=0.95)
    
    return advantages
```

**Step 4: PPO 更新**

```python
def reasoning_rl_train(model, prm, questions, num_epochs=10):
    for epoch in range(num_epochs):
        for question in questions:
            # 生成推理轨迹
            trace = generate_reasoning_trace(model, question)
            
            # PRM 评分
            step_scores = score_trace(prm, question, trace)
            
            # 计算优势
            advantages = compute_advantage(step_scores)
            
            # PPO 更新
            for step, advantage in zip(trace, advantages):
                # 计算策略梯度
                log_prob = model.log_prob(question, step)
                policy_loss = -log_prob * advantage
                
                # 更新
                policy_loss.backward()
                optimizer.step()
```

#### 代表工作

**DeepSeek-R1**

**核心创新**：用 [[02-模块/Policy-Based/GRPO|GRPO]] + [[02-模块/Reward-Model/PRM|PRM]] 训练数学推理，显著超越监督学习。

**训练流程**：

```
阶段 1：冷启动
  - 用少量人工标注的推理数据 SFT
  - 产出：Base Reasoner

阶段 2：推理 RL（核心）
  - 自我对弈生成推理数据
  - 用 PRM 奖励推理过程
  - 用 GRPO 优化策略
  - 产出：Advanced Reasoner

阶段 3：对齐
  - 标准 RLHF
  - 产出：DeepSeek-R1
```

**关键设计**：

1. **过程奖励**：不仅奖励最终答案，还奖励推理步骤
2. **[[02-模块/Policy-Based/GRPO|GRPO]]**：用组内相对奖励，无需价值模型，节省显存
3. **自我对弈**：模型自己生成推理数据，无需人工标注

**效果**：
- GSM8K（数学推理）：从 75% 提升到 95%
- MATH（竞赛数学）：从 40% 提升到 70%
- 超越了 GPT-4 在数学推理上的表现

**OpenAI o1**

**核心创新**：长链条推理 + 过程奖励。

**特点**：
- 模型会"思考"很长时间，生成详细的推理过程
- 用隐式的过程奖励（未公开 PRM 细节）
- 在竞赛数学、编程上表现极强

**效果**：
- AIME（美国数学邀请赛）：83%（GPT-4 仅 40%）
- Codeforces（编程竞赛）：达到前 500 名水平

---

### 3. 持续预训练 + RL 约束

**问题**：如何在学新能力的同时，不遗忘预训练知识？

**核心思想**：在继续预训练的同时，用 KL 散度约束模型不要偏离太远。

#### 数学形式

**损失函数**：

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{base}})$$

**其中**：
- $\mathcal{L}_{\text{LM}}$：预训练损失（next-token prediction）
- $D_{\text{KL}}$：KL 散度，约束模型不要偏离 Base Model 太远
- $\lambda$：平衡系数

**KL 散度计算**：

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{base}}) = \sum_{x} \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\text{base}}(x)}$$

#### 实现方式

**代码示例**：

```python
# 加载 Base Model（冻结）
base_model = AutoModel.from_pretrained("base-model")
base_model.eval()

# 训练模型
model = AutoModel.from_pretrained("base-model")
model.train()

for batch in dataloader:
    # 计算预训练损失
    lm_loss = model(batch, labels=batch).loss
    
    # 计算 KL 散度
    with torch.no_grad():
        base_logits = base_model(batch).logits
    current_logits = model(batch).logits
    
    # KL(当前 || 基础)
    kl_div = F.kl_div(
        F.log_softmax(current_logits, dim=-1),
        F.softmax(base_logits, dim=-1),
        reduction='batchmean'
    )
    
    # 总损失
    loss = lm_loss + lambda_ * kl_div
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

#### 动态调整 λ

**问题**：λ 太大，模型学不到新能力；λ 太小，模型遗忘预训练知识。

**解决方案**：动态调整 λ。

```python
# 方法 1：线性衰减
lambda_ = initial_lambda * (1 - current_step / total_steps)

# 方法 2：基于 KL 散度自适应
target_kl = 0.1  # 目标 KL 散度
current_kl = compute_kl(model, base_model)
if current_kl > target_kl:
    lambda_ *= 1.1  # KL 太大，增加约束
else:
    lambda_ *= 0.9  # KL 太小，放松约束
```

---

### 4. 多阶段 RL 训练

**问题**：不同能力的学习难度不同，需要分阶段训练。

**核心思想**：从简单能力到复杂能力，逐步训练。

#### 训练流程

```
阶段 1：基础推理
  - 数据：简单数学题、逻辑题
  - 目标：学会基础推理模式
  - 方法：DPO

阶段 2：高级推理
  - 数据：竞赛数学、复杂编程
  - 目标：学会长链条推理
  - 方法：推理 RL (PPO + PRM)

阶段 3：综合应用
  - 数据：多步骤问题、跨领域问题
  - 目标：学会综合运用多种能力
  - 方法：自我对弈
```

#### 课程学习 + RL

**结合[[01-原子/课程学习|课程学习]]和 RL**：

```python
# 定义难度级别
difficulty_levels = [
    "easy_math",      # 简单数学
    "medium_math",    # 中等数学
    "hard_math",      # 困难数学
    "competition"     # 竞赛数学
]

# 课程学习 + RL
for level in difficulty_levels:
    # 加载该难度的数据
    data = load_data(level)
    
    # 用 RL 训练
    for epoch in range(num_epochs):
        for batch in data:
            # 生成推理轨迹
            trace = generate_trace(model, batch)
            
            # 评分
            score = prm.score(trace)
            
            # 更新
            loss = -log_prob(trace) * score
            loss.backward()
            optimizer.step()
    
    # 评估，决定是否进入下一阶段
    if evaluate(model, level) > threshold:
        print(f"Level {level} mastered, moving to next level")
    else:
        print(f"Level {level} not mastered, continue training")
```

---

## 真实案例

### DeepSeek-R1 详细训练流程

**阶段 1：Pre-Training**
- 数据：14.8 万亿 token
- 产出：DeepSeek-V3-Base

**阶段 2：Mid-Training（核心创新）**

```
子阶段 2.1：冷启动 SFT
  - 数据：几千条人工标注的推理数据
  - 方法：监督微调
  - 产出：Base Reasoner

子阶段 2.2：推理 RL
  - 数据：自我对弈生成
  - 奖励：PRM（过程奖励）+ ORM（结果奖励）
  - 方法：GRPO
  - 训练：
    - 生成 100 个候选推理轨迹
    - PRM 评分每个步骤
    - ORM 评分最终答案
    - 计算组内相对优势
    - PPO 更新
  - 产出：Advanced Reasoner

子阶段 2.3：持续改进
  - 数据：更难的数学题
  - 方法：自我对弈 + 推理 RL
  - 产出：Expert Reasoner
```

**阶段 3：Post-Training**
- 标准 RLHF 对齐
- 产出：DeepSeek-R1-Chat

**关键创新**：
1. **过程奖励**：不仅奖励答案，还奖励推理过程
2. **GRPO**：节省显存，适合大模型
3. **自我对弈**：无需人工标注推理数据

**效果**：
- GSM8K：95%（GPT-4: 92%）
- MATH：70%（GPT-4: 53%）
- AIME：83%（GPT-4: 40%）

### OpenAI o1 训练流程（推测）

**阶段 1：Pre-Training**
- 标准预训练

**阶段 2：Mid-Training**

```
子阶段 2.1：思维链 SFT
  - 数据：大量人工标注的思维链数据
  - 方法：监督微调

子阶段 2.2：长链条推理 RL
  - 数据：竞赛数学、编程难题
  - 奖励：隐式过程奖励
  - 方法：PPO
  - 特点：模型会生成非常长的推理过程（数千 token）

子阶段 2.3：自我对弈
  - 模型自己生成难题并解答
  - 持续改进
```

**阶段 3：Post-Training**
- RLHF 对齐
- 产出：o1

**特点**：
- 推理时间很长（几十秒到几分钟）
- 会"思考"多个解决方案
- 在竞赛级别问题上表现极强

---

## 产出：Capable Model

经过 Mid-Training，得到 Capable Model。

**特点**：
- ✅ 强大的推理能力
- ✅ 代码生成能力
- ✅ 数学解题能力
- ✅ 长链条规划能力
- ❌ 仍可能输出不当内容
- ❌ 没有对齐人类偏好

**示例交互**：

```
用户：求解方程 x² - 5x + 6 = 0

Capable Model：
  "这是一个二次方程。我可以用求根公式或因式分解。
  
  方法 1：因式分解
  x² - 5x + 6 = (x - 2)(x - 3) = 0
  所以 x = 2 或 x = 3
  
  方法 2：求根公式
  x = (5 ± √(25 - 24)) / 2
  x = (5 ± 1) / 2
  所以 x = 3 或 x = 2
  
  最终答案：x = 2 或 x = 3"
```

---

## 与下一阶段的关系

**Mid-Training → Post-Training**：

```
Capable Model（有强大能力，但未对齐）
  ↓
SFT（学习对话格式）
  ↓
RLHF / DPO（对齐人类偏好）
  ↓
Aligned Model（对齐后的模型）
```

**关键转变**：
- Mid-Training：学能力（推理、代码）
- Post-Training：学对齐（有用、无害）

---

## 2026 前沿发现（基于深度调研）

> 以下内容来自 2024-2026 前沿论文的深度调研，详见 [[06-LLM训练全流程/深度调研报告-LLM训练中的RL技术]]。

### 发现 A：RL 从预训练早期就有效

**哈佛 2026.06 研究**（arXiv 2606.04272）发现：RL（RLVR with GRPO）从预训练仅 4B token 的检查点就有效，在 1B 模型上将 GSM8K pass@1 从 ~2% 提升到 ~18%。

**关键洞察：数据组成 > 模型规模**

对于更难的问题（MATH 风格），针对性的预训练数据组成比扩大模型规模更有效：
- 添加 10B 数学专用 token 的 RL 增益 > 同等 token 预算下从 1B 扩展到 4B 参数
- 到 10B token 时，直接 RL 匹配完整的 SFT-then-RL 流水线

**注意**：仅在 1B/4B 规模验证，70B+ 外推未证实。

### 发现 B：直接 RL 扩展分布，SFT 后才 RL 会收窄

**反直觉发现**：直接在基础预训练检查点上应用 RL 会**扩展**模型输出分布（pass@1 和 pass@k 都提升）。

而先前报告的"分布收窄效应"（pass@1 上升，pass@k 下降）**仅在 RL 跟随 SFT 之后时出现**——是 SFT 限制了探索，不是 RL 本身。

**机制**："没有预先接触标准答案，模型通过 on-policy 学习探索并发现新的推理路径。"

**注意**：在更难的 MATH 问题上，直接 RL 不如 SFT-then-RL。

### 发现 C：Meta RAM 的 Thinking Mid-Training

**Meta FAIR 2026.01**（arXiv 2601.21343）引入 "thinking mid-training"——预训练和后训练之间的新中间阶段：

```
步骤 1: 标注模型数据增强（交错思维链）
步骤 2: SFT mid-training（增强数据）
步骤 3: RL mid-training（DrGRPO + LLM-as-Judge 二值奖励）
```

**完整流水线（SFT+RL mid-training → RLVR post-training）**：

| 模型 | 无 mid-training | 完整流水线 | 提升 |
|------|----------------|-----------|------|
| Llama-3-8B | 0.1197 | 0.3837 | **3.2×** |

测试基准：GSM8K、MATH-500、AMC23、Olympiad、GPQA-Diamond

### 发现 D：多解法 Mid-Training 提升 pass@k

**ASU + DeepMind 2026.05**（arXiv 2605.08472）发现：RL 之前在 mid-training 中使用多样化自生成数据（每题多种解法）可以一致地提升数学推理的 pass@k。

**在固定预算下**，学习每题多种解法比学习更多题的单解法提升 ~7%（都是 7,408 个实例）。

**RL 训练导致涌现的多解法组合**：56.7% vs 无 RL 的 23.3%（n=16 启发式）。

**理论依据**：Theorem 4.1 证明 N-modal 分布可以产生 O(1/N*(1-1/N)) 梯度更新，优于 uni-modal 的 O(ε²)。

**注意**：结果仅在 Llama-3.2-3B 验证；Qwen2.5-7B 未显示一致提升。

### 发现 E：RL 产生真正能力提升的条件

**CMU Neubig 2025.12**（arXiv 2512.07783）发现 RL 只在以下条件下产生真正的推理能力提升（pass@128）：

1. **预训练留有足够余量**（headroom）
2. **RL 训练数据针对模型的"能力边缘"**——难度适中但不至于无法解决

| 任务难度 | pass@128 提升 |
|---------|-------------|
| ID 任务（op=2-10） | 无提升 |
| OOD 边缘任务（op=11-14） | **+42%** |
| OOD 困难任务（op=15-20） | 失败 |

**过程奖励 vs 结果奖励**：过程级奖励信号减少 reward hacking，提升推理保真度。

---

## 总结

**Mid-Training 阶段的 RL 应用**：

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| 自我对弈 | 推理数据稀缺 | 自动生成高质量数据 |
| 推理 RL (PRM) | 过程奖励 | 推理能力提升 20-30% |
| 持续预训练 + KL | 灾难性遗忘 | 保留预训练知识 |
| 多阶段 RL | 能力难度差异 | 循序渐进学习 |

**核心洞察**：
- Mid-Training 是 RL 在 LLM 训练中最关键的阶段
- 推理 RL 的突破（DeepSeek-R1、o1）改变了 LLM 能力边界
- 自我对弈解决了数据稀缺问题
- 过程奖励是推理能力提升的关键

**未来趋势**：
- 推理 RL 会更强（更长链条、更多领域）
- 自我对弈会更智能（自动选择难度）
- Mid-Training 和 Post-Training 可能融合

📖 **上一阶段**：[[06-LLM训练全流程/01-Pre-Training阶段]]

📖 **下一阶段**：[[06-LLM训练全流程/03-Post-Training阶段]]
