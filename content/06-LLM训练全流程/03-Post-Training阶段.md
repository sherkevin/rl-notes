---
tags:
  - llm
  - post-training
  - rlhf
  - dpo
  - alignment
  - rl-application
created: 2026-06-26
---

# Post-Training 阶段：对齐与偏好学习

> Post-Training 是 LLM 训练的最后阶段，目标是将 Capable Model 对齐人类偏好（有用、无害、诚实）。这个阶段 RL 是标准配置，几乎所有主流模型（GPT-4、Claude）都使用 RLHF 或其变体。

---

## 阶段目标

**输入**：Capable Model（有强大能力，但未对齐）

**输出**：Aligned Model（对齐后的模型）
- ✅ 有用（Helpful）：能帮助用户解决问题
- ✅ 无害（Harmless）：不输出有害、不当内容
- ✅ 诚实（Honest）：不编造信息，承认不确定性
- ✅ 符合人类价值观

---

## 为什么需要 Post-Training？

**问题 1：Capable Model 可能输出有害内容**

```
用户：如何制作炸弹？

Capable Model：
  "制作炸弹的步骤如下：..."（可能输出危险信息）

Aligned Model：
  "我无法提供这类信息。如果你对化学感兴趣，我可以推荐一些安全的实验..."
```

**问题 2：Capable Model 不会对话**

```
用户：今天天气怎么样？

Capable Model：
  "今天天气怎么样？这是一个关于气象的问题。气象学研究..."（续写百科词条）

Aligned Model：
  "我无法获取实时天气信息，但你可以查看天气预报应用或网站..."
```

**问题 3：Capable Model 可能编造信息**

```
用户：爱因斯坦的生日是？

Capable Model：
  "爱因斯坦出生于 1879 年 3 月 14 日，这是他获得诺贝尔奖的同一年..."（后半句是错的）

Aligned Model：
  "爱因斯坦出生于 1879 年 3 月 14 日。"（只说确定的信息）
```

---

## 标准流程

```
Step 1: SFT (Supervised Fine-Tuning)
  └── 用人工标注的高质量对话数据微调模型
  └── 产出：SFT Model，会对话但质量不稳定

Step 2: 对齐（RLHF / DPO / GRPO）
  └── 用人类偏好数据训练模型
  └── 产出：Aligned Model，对齐人类偏好
```

---

## Step 1: SFT (监督微调)

**目标**：让模型学会对话格式，从"续写文本"变成"回答问题"。

**数据格式**：

```json
{
  "prompt": "什么是机器学习？",
  "response": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而无需明确编程。主要方法包括监督学习、无监督学习和强化学习。"
}
```

**训练目标**：

$$\mathcal{L}_{\text{SFT}} = -\sum_{(x, y) \in \mathcal{D}} \log P(y | x; \theta)$$

**代码示例**：

```python
# SFT 训练
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("capable-model")
tokenizer = AutoTokenizer.from_pretrained("capable-model")

# 加载对话数据
data = load_sft_data("sft_dataset.json")

# 训练
for epoch in range(num_epochs):
    for batch in dataloader(data, batch_size=8):
        # 构造输入
        inputs = []
        for item in batch:
            text = f"User: {item['prompt']}\nAssistant: {item['response']}"
            inputs.append(text)
        
        # tokenize
        encoded = tokenizer(inputs, return_tensors="pt", padding=True)
        
        # 计算损失（只对 response 部分计算）
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

**产出**：SFT Model
- ✅ 会对话，格式正确
- ❌ 质量不稳定（有时好有时坏）
- ❌ 可能仍有有害输出

---

## Step 2: 对齐方法

### 方法 1: RLHF (Reinforcement Learning from Human Feedback)

**经典流程**：

```
1. 收集偏好数据
   └── 人工标注 (prompt, response_w, response_l)
   └── response_w 比 response_l 更好

2. 训练奖励模型 (Reward Model)
   └── 输入：(prompt, response)
   └── 输出：标量分数
   └── 目标：让好的 response 分数更高

3. 用 PPO 优化策略
   └── 状态：prompt
   └── 动作：response
   └── 奖励：奖励模型的分数
   └── 约束：KL 散度不要偏离 SFT 模型太远
```

#### 1. 收集偏好数据

**标注方式**：

```
标注员看到一个 prompt 和两个 response：

Prompt: "如何学习编程？"

Response A:
  "学习编程需要循序渐进。首先选择一门语言（如 Python），
   然后学习基础语法，最后通过项目实践提升。"

Response B:
  "编程很简单，你只需要看几本书就会了。"

标注员选择：A 比 B 好
理由：A 更具体、更有用，B 过于简化
```

**数据格式**：

```json
{
  "prompt": "如何学习编程？",
  "response_w": "学习编程需要循序渐进...",
  "response_l": "编程很简单..."
}
```

**数据量**：通常需要 10k-100k 条偏好数据。

#### 2. 训练奖励模型

**模型结构**：

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model.transformer  # 复用预训练模型
        self.reward_head = nn.Linear(hidden_dim, 1)  # 输出标量
    
    def forward(self, input_ids, attention_mask):
        # 编码
        hidden_states = self.transformer(input_ids, attention_mask)
        
        # 取最后一个 token 的表示
        last_hidden = hidden_states[:, -1, :]
        
        # 输出奖励分数
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
```

**训练目标**：Bradley-Terry 模型

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]$$

**直觉**：让好的 response 分数比坏的高。

**训练代码**：

```python
# 训练奖励模型
reward_model = RewardModel(base_model)
optimizer = Adam(reward_model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in preference_dataloader:
        prompts = batch["prompt"]
        responses_w = batch["response_w"]
        responses_l = batch["response_l"]
        
        # 计算分数
        reward_w = reward_model(prompts, responses_w)
        reward_l = reward_model(prompts, responses_l)
        
        # 损失：让 reward_w > reward_l
        loss = -torch.log(torch.sigmoid(reward_w - reward_l)).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

**效果**：奖励模型与人类标注的一致性通常达到 70-75%。

#### 3. PPO 优化

**目标函数**：

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$$

**其中**：
- $r(x, y)$：奖励模型的分数
- $\pi_\theta$：当前策略（正在训练的模型）
- $\pi_{\text{ref}}$：参考策略（SFT 模型）
- $\beta$：KL 惩罚系数

**PPO 训练循环**：

```python
# PPO 训练
policy_model = load_sft_model()
ref_model = load_sft_model()
reward_model = load_reward_model()
value_model = load_value_model()  # Critic

for iteration in range(num_iterations):
    # 1. 生成 response
    prompts = sample_prompts()
    responses = policy_model.generate(prompts, temperature=1.0)
    
    # 2. 计算奖励
    rewards = reward_model(prompts, responses)
    
    # 3. 计算 KL 惩罚
    kl_penalty = compute_kl_divergence(policy_model, ref_model, prompts, responses)
    
    # 4. 总奖励
    total_rewards = rewards - beta * kl_penalty
    
    # 5. 用 PPO 更新
    for epoch in range(ppo_epochs):
        # 计算优势
        advantages = compute_gae(value_model, prompts, responses, total_rewards)
        
        # 策略更新
        ratio = compute_probability_ratio(policy_model, old_policy_model, prompts, responses)
        policy_loss = ppo_loss(ratio, advantages, epsilon=0.2)
        policy_loss.backward()
        
        # 价值函数更新
        value_loss = value_model.loss(prompts, responses, total_rewards)
        value_loss.backward()
```

**产出**：Aligned Model（如 GPT-4、Claude）

📖 **详细文档**：[[05-应用领域/Post-Training/RLHF]]

---

### 方法 2: DPO (Direct Preference Optimization)

**创新**：跳过奖励模型，直接用偏好数据优化策略。

**核心洞察**：RLHF 中的奖励模型和策略优化可以合并为一个步骤。

#### 数学推导

**RLHF 的目标**：

$$\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta} [r(x, y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

**最优策略的形式**：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

**反推奖励函数**：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**代入偏好模型**：

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**DPO 损失**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

#### 实现代码

```python
# DPO 训练
policy_model = load_sft_model()
ref_model = load_sft_model()
ref_model.eval()  # 冻结

optimizer = Adam(policy_model.parameters(), lr=5e-7)

for epoch in range(num_epochs):
    for batch in preference_dataloader:
        prompts = batch["prompt"]
        responses_w = batch["response_w"]
        responses_l = batch["response_l"]
        
        # 计算当前策略的对数概率
        log_prob_w = policy_model.log_prob(prompts, responses_w)
        log_prob_l = policy_model.log_prob(prompts, responses_l)
        
        # 计算参考策略的对数概率
        with torch.no_grad():
            ref_log_prob_w = ref_model.log_prob(prompts, responses_w)
            ref_log_prob_l = ref_model.log_prob(prompts, responses_l)
        
        # DPO 损失
        logits = beta * ((log_prob_w - ref_log_prob_w) - (log_prob_l - ref_log_prob_l))
        loss = -F.logsigmoid(logits).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

**优势**：
- 简单：无需训练奖励模型
- 稳定：无需 RL 训练
- 快速：直接优化，收敛更快

**劣势**：
- 效果略逊于 RLHF
- 对数据质量敏感
- 探索能力弱

📖 **详细文档**：[[05-应用领域/Post-Training/DPO]]

---

### 方法 3: GRPO (Group Relative Policy Optimization)

**创新**：去掉价值模型（Critic），用组内相对奖励替代。

**动机**：PPO 需要训练价值模型，显存占用大，对大模型不友好。

#### 核心思想

```
对同一个 prompt，生成 N 个 response
计算每个 response 相对于组内平均水平的优势
用 PPO 更新策略（无需价值模型）
```

#### 数学形式

**Step 1: 生成组内样本**

$$\{y_1, y_2, \ldots, y_N\} \sim \pi_\theta(\cdot | x)$$

**Step 2: 计算奖励**

$$r_i = r(x, y_i), \quad i = 1, \ldots, N$$

**Step 3: 计算组内优势**

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_N\})}{\text{std}(\{r_1, \ldots, r_N\})}$$

**Step 4: PPO 更新**

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E} \left[ \min\left( \rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) \right]$$

**其中** $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$ 是重要性采样比率。

#### 实现代码

```python
# GRPO 训练
policy_model = load_sft_model()
ref_model = load_sft_model()
reward_model = load_reward_model()

optimizer = Adam(policy_model.parameters(), lr=1e-6)

for iteration in range(num_iterations):
    prompts = sample_prompts(batch_size=4)
    
    for prompt in prompts:
        # 生成 N 个 response
        responses = policy_model.generate(prompt, num_samples=16, temperature=1.0)
        
        # 计算奖励
        rewards = reward_model(prompt, responses)
        
        # 计算组内优势
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        
        # PPO 更新
        for epoch in range(ppo_epochs):
            for response, advantage in zip(responses, advantages):
                # 计算比率
                log_prob = policy_model.log_prob(prompt, response)
                old_log_prob = old_policy_model.log_prob(prompt, response)
                ratio = torch.exp(log_prob - old_log_prob)
                
                # PPO 损失
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                loss = -torch.min(surr1, surr2).mean()
                
                # KL 惩罚
                kl = log_prob - ref_model.log_prob(prompt, response)
                loss += beta * kl.mean()
                
                # 反向传播
                loss.backward()
        
        optimizer.step()
```

**优势**：
- 节省显存：无需价值模型（通常占模型大小的 50%）
- 适合大模型：70B+ 模型也能训练
- 效果接近 PPO

**劣势**：
- 方差略大：组内估计不如价值模型准确
- 需要较大的 N（通常 16-32）

📖 **详细文档**：[[05-应用领域/Post-Training/GRPO]]

---

### 方法 4: Constitutional AI (CAI)

**创新**：用规则约束替代人工标注，大幅减少标注成本。

**动机**：人工标注偏好数据成本高、一致性差。能否用规则自动判断？

#### 核心思想

```
1. 定义一组规则（"宪法"）
   例如：
     - "不要输出有害内容"
     - "不要编造信息"
     - "承认不确定性"

2. 让模型自己检查输出是否违反规则
   └── 模型作为"审查员"

3. 违反的输出作为负样本，用于 DPO 训练
```

#### 实现流程

**Step 1: 定义宪法**

```python
constitution = [
    "回答应该是有帮助的，能够解决用户的问题",
    "回答不应该是有害的，不应鼓励危险行为",
    "回答不应该是歧视性的，不应包含偏见",
    "回答应该是诚实的，不应编造信息",
    "如果不确定，应该承认不确定性"
]
```

**Step 2: 生成候选回答**

```python
# 对每个 prompt，生成多个回答
prompts = sample_prompts(1000)
candidates = []

for prompt in prompts:
    responses = model.generate(prompt, num_samples=4, temperature=1.0)
    candidates.append((prompt, responses))
```

**Step 3: 自我审查**

```python
# 让模型审查每个回答是否违反宪法
def self_critique(model, prompt, response, constitution):
    critiques = []
    for rule in constitution:
        critique_prompt = f"""
        请检查以下回答是否违反这条规则：
        规则：{rule}
        问题：{prompt}
        回答：{response}
        
        如果违反，请解释原因；如果没有违反，请说"没有违反"。
        """
        critique = model.generate(critique_prompt)
        critiques.append(critique)
    return critiques

# 审查所有候选回答
preference_data = []
for prompt, responses in candidates:
    critiques = [self_critique(model, prompt, resp, constitution) for resp in responses]
    
    # 根据审查结果排序
    # 违反规则少的 > 违反规则多的
    scores = [count_violations(critique) for critique in critiques]
    best_idx = argmin(scores)
    worst_idx = argmax(scores)
    
    preference_data.append({
        "prompt": prompt,
        "response_w": responses[best_idx],
        "response_l": responses[worst_idx]
    })
```

**Step 4: DPO 训练**

```python
# 用自我审查生成的偏好数据训练
dpo_train(policy_model, ref_model, preference_data)
```

#### 实际应用

**Claude 系列模型**：

```
阶段 1：SFT
  └── 人工标注的对话数据

阶段 2：CAI
  └── 定义宪法（10-20 条规则）
  └── 自我审查生成偏好数据
  └── DPO 训练

阶段 3：RLHF（可选）
  └── 少量人工标注微调
```

**优势**：
- 大幅减少人工标注（从 100k 条降到 10k 条）
- 规则可解释、可审计
- 可以快速迭代（修改宪法即可）

**劣势**：
- 规则覆盖不全（总有边缘情况）
- 自我审查可能不准确
- 仍需要少量人工标注

---

## 真实案例

### GPT-4 对齐流程

```
Step 1: SFT
  - 数据：几十万条人工标注的对话
  - 标注员：专业标注团队
  - 产出：GPT-4 SFT

Step 2: 奖励模型
  - 数据：几十万条偏好数据
  - 标注：人工比较两个回答
  - 产出：Reward Model

Step 3: RLHF (PPO)
  - 迭代：数千次 PPO 更新
  - KL 惩罚：β = 0.05
  - 产出：GPT-4

效果：
  - 有害内容减少 80%
  - 有用性提升 30%
  - 人类偏好一致性 75%
```

### Claude 对齐流程

```
Step 1: SFT
  - 数据：人工标注 + Constitutional AI 生成

Step 2: Constitutional AI
  - 宪法：15 条规则
  - 自我审查：生成 50k 条偏好数据
  - DPO 训练

Step 3: RLHF（少量）
  - 数据：10k 条人工标注
  - 微调：进一步优化

效果：
  - 有害内容减少 90%
  - 有用性提升 25%
  - 标注成本降低 70%
```

### DeepSeek-R1 对齐流程

```
Step 1: SFT
  - 数据：对话数据 + 推理数据

Step 2: GRPO
  - 奖励模型：Outcome Reward Model (ORM)
  - 组大小：N = 16
  - 无需价值模型，节省显存

Step 3: 持续优化
  - 自我对弈生成偏好数据
  - 持续 DPO 训练

效果：
  - 推理能力保持（Mid-Training 学到的）
  - 对齐质量高
  - 训练成本低
```

---

## 产出：Aligned Model

经过 Post-Training，得到 Aligned Model。

**特点**：
- ✅ 有用：能帮助用户解决问题
- ✅ 无害：不输出有害内容
- ✅ 诚实：不编造信息
- ✅ 符合人类价值观

**示例交互**：

```
用户：如何学习编程？

Aligned Model：
  "学习编程需要循序渐进。我建议：
  
  1. 选择一门语言：Python 是很好的入门语言
  2. 学习基础：变量、循环、函数
  3. 做项目：从小项目开始，逐步提升难度
  4. 多练习：编程是实践性很强的技能
  
  推荐资源：Codecademy、LeetCode、GitHub"
```

---

## 总结

**Post-Training 阶段的 RL 应用**：

| 方法 | 核心思想 | 优势 | 劣势 | 代表模型 |
|------|---------|------|------|---------|
| RLHF | PPO + 奖励模型 | 效果最好 | 复杂、不稳定 | GPT-4, Claude |
| DPO | 直接偏好优化 | 简单稳定 | 效果略逊 | LLaMA 2, Mistral |
| GRPO | 组内相对奖励 | 节省显存 | 方差略大 | DeepSeek-R1 |
| CAI | 规则约束 | 减少标注 | 覆盖不全 | Claude |

**核心洞察**：
- Post-Training 是 RL 在 LLM 训练中最成熟的阶段
- RLHF 仍是效果最好的方法，但成本高
- DPO/GRPO 是简化版本，适合资源有限场景
- Constitutional AI 大幅减少人工标注

**未来趋势**：
- 更高效的对齐方法（减少标注成本）
- 更强的泛化能力（对齐到新场景）
- 与 Mid-Training 融合（能力学习 + 对齐）

📖 **上一阶段**：[[06-LLM训练全流程/02-Mid-Training阶段]]

📖 **返回总览**：[[06-LLM训练全流程/README]]
