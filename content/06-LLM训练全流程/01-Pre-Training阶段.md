---
tags:
  - llm
  - pre-training
  - rl-application
  - curriculum-learning
created: 2026-06-26
---

# Pre-Training 阶段：基础模型训练

> 预训练是 LLM 训练的第一步，目标是从海量文本学习语言模式和世界知识。虽然主要使用自监督学习，但 RL 技术正在逐步渗透，用于提高数据效率。

---

## 阶段目标

**输入**：海量文本语料（数万亿 token）

**输出**：Base Model（基础模型）
- 具备语言理解能力
- 掌握世界知识
- 有基础推理能力
- **但不会对话，可能输出有害内容**

---

## 传统方法：Next-Token Prediction

**核心思想**：预测下一个 token，不断预测就学会了语言。

$$\mathcal{L}_{\text{LM}} = -\sum_{i=1}^{N} \log P(x_i | x_{<i}; \theta)$$

**其中**：
- $x_i$：第 $i$ 个 token
- $x_{<i}$：前 $i-1$ 个 token（上下文）
- $\theta$：模型参数
- $P(x_i | x_{<i}; \theta)$：模型预测的概率

**训练过程**：

```
输入："今天天气"
目标：预测 "很好"

模型输出：P("很") = 0.3, P("好") = 0.2, ...
损失：-log(0.3) - log(0.2)
```

**优势**：
- 无需人工标注，数据无限
- 可以学到丰富的语言模式
- 扩展性好（数据越多，模型越大，效果越好）

**局限**：
- 数据质量参差不齐
- 随机采样效率低
- 可能学到有害内容

---

## RL 技术应用

### 1. 课程学习 (Curriculum Learning)

**问题**：数据难度分布不均，随机采样导致模型在简单数据上过度训练，在难数据上训练不足。

**RL 方法**：用 RL 学习数据采样策略，动态调整数据难度。

#### 状态空间

```
状态 = (模型当前性能, 各类数据的 loss, 训练进度)
```

#### 动作空间

```
动作 = 选择下一批数据的来源和难度
例如：
  - 从 Wikipedia 采样 50%
  - 从学术论文采样 30%
  - 从代码库采样 20%
```

#### 奖励设计

```
奖励 = 模型性能提升速度
例如：
  - 在验证集上 perplexity 下降速度
  - 在下游任务上的准确率提升
```

#### 实现方式

**方法 1：基于规则的启发式**

```python
# 简单规则：优先采样 loss 高的数据
def sample_data(model, data_sources):
    losses = {src: model.compute_loss(src) for src in data_sources}
    # 优先采样 loss 高的（模型不擅长的）
    probs = softmax([losses[src] for src in data_sources])
    return random_choice(data_sources, probs)
```

**方法 2：多臂老虎机 (Multi-Armed Bandit)**

```python
# 把每个数据源当作一个"老虎机"
# 用 UCB (Upper Confidence Bound) 策略平衡探索和利用
def ucb_select(rewards, counts, total_steps, c=2.0):
    ucb_scores = []
    for i in range(len(rewards)):
        if counts[i] == 0:
            return i  # 优先尝试未试过的
        mean = rewards[i] / counts[i]
        bonus = c * sqrt(log(total_steps) / counts[i])
        ucb_scores.append(mean + bonus)
    return argmax(ucb_scores)
```

**方法 3：深度 RL**

```python
# 用 DQN 或 PPO 学习采样策略
state = (model_embeddings, data_features, training_progress)
action = data_source_index
reward = performance_improvement

# 训练一个小的 RL agent
agent = DQN(state_dim, action_dim)
for step in range(total_steps):
    state = get_current_state()
    action = agent.select_action(state)
    batch = sample_from_source(action)
    model.train_on_batch(batch)
    reward = evaluate_improvement()
    agent.update(state, action, reward)
```

#### 实际应用

**GPT-3 的课程学习**：
- 早期：主要采样简单文本（儿童读物、维基百科）
- 中期：增加复杂文本（新闻、书籍）
- 后期：采样学术论文、代码

**效果**：相比随机采样，收敛速度提升 20-30%。

📖 **相关原子**：[[01-原子/课程学习]]

---

### 2. 数据选择 RL

**问题**：不是所有数据都有价值。有些数据重复、低质、甚至有害。

**RL 方法**：训练一个"数据选择器"，学习哪些数据对模型最有帮助。

#### 状态空间

```
状态 = 数据特征向量
例如：
  - 文本长度
  - 词汇丰富度
  - 领域标签（科技、娱乐、新闻...）
  - 重复度（与已有数据的相似度）
```

#### 动作空间

```
动作 ∈ {保留, 丢弃, 降低权重}
```

#### 奖励设计

```
奖励 = 使用该数据后模型在验证集上的性能提升
例如：
  - 保留高质量数据 → perplexity 下降 → 正奖励
  - 保留重复数据 → perplexity 不变 → 零奖励
  - 保留有害数据 → 安全评估下降 → 负奖励
```

#### 实现方式

**两阶段训练**：

```
阶段 1：训练数据选择器
  1. 随机采样一小部分数据，训练基础模型
  2. 用基础模型评估剩余数据的价值
  3. 训练 RL agent 学习选择策略

阶段 2：用选择器筛选数据
  1. 用训练好的选择器筛选全量数据
  2. 用筛选后的数据重新训练模型
```

**代码示例**：

```python
# 数据选择器
class DataSelector:
    def __init__(self):
        self.policy = PPO(
            state_dim=data_feature_dim,
            action_dim=3,  # 保留/丢弃/降权
        )
    
    def select(self, data_batch):
        features = extract_features(data_batch)
        actions = self.policy.act(features)
        weights = map_action_to_weight(actions)
        return weights  # 每个样本的权重

# 训练循环
selector = DataSelector()
for epoch in range(num_epochs):
    data_batch = sample_random_data()
    weights = selector.select(data_batch)
    
    # 用权重训练模型
    model.train(data_batch, sample_weights=weights)
    
    # 评估性能
    reward = evaluate_model(model)
    
    # 更新选择器
    selector.policy.update(reward)
```

#### 实际应用

**Dolma 数据集（Allen AI）**：
- 用启发式规则 + 小模型评估数据质量
- 过滤掉重复、低质、有害数据
- 相比未过滤数据，模型性能提升 15%

**RedPajama 数据集（Cerebras）**：
- 用分类器打分，过滤低质数据
- 用 RL 优化分类器阈值

---

### 3. 数据去重与多样性 RL

**问题**：数据中存在大量重复，导致模型过拟合；同时需要保证数据多样性。

**RL 方法**：学习去重策略，在去重和保留信息之间取得平衡。

#### 状态空间

```
状态 = (当前数据相似度矩阵, 已选数据量, 目标数据量)
```

#### 动作空间

```
动作 = 对每对相似数据，决定保留哪个或都保留
```

#### 奖励设计

```
奖励 = -λ₁·重复度 - λ₂·信息损失
重复度：选中数据之间的平均相似度
信息损失：被丢弃数据的信息量
```

#### 实现方式

**基于 MinHash 的快速去重**：

```python
# 计算文本的 MinHash 签名
def minhash(text, num_hashes=128):
    hashes = []
    for hash_func in hash_functions[:num_hashes]:
        hashes.append(min(hash_func(shingle) for shingle in get_shingles(text)))
    return hashes

# 计算 Jaccard 相似度
def jaccard_similarity(hash1, hash2):
    return sum(h1 == h2 for h1, h2 in zip(hash1, hash2)) / len(hash1)

# RL agent 决定去重策略
agent = PPO(state_dim, action_dim)
for batch in data_batches:
    similarities = compute_pairwise_similarities(batch)
    state = (similarities, selected_count, target_count)
    actions = agent.act(state)  # 对每对决定保留/丢弃
    reward = compute_reward(actions, similarities)
    agent.update(state, actions, reward)
```

---

## 产出：Base Model

经过预训练（可能包含 RL 优化），得到 Base Model。

**特点**：
- ✅ 语言流畅，语法正确
- ✅ 掌握丰富知识
- ✅ 有基础推理能力
- ❌ 不会对话（只会续写文本）
- ❌ 可能输出有害内容
- ❌ 没有特定领域的深度能力

**示例交互**：

```
用户：今天天气怎么样？
Base Model：今天天气怎么样？这是一个很好的问题。天气是指...（开始续写百科词条）

期望：
用户：今天天气怎么样？
Aligned Model：我无法获取实时天气信息，但你可以查看天气预报应用...
```

---

## 真实案例

### LLaMA 2 预训练

**数据**：2 万亿 token

**[[01-原子/课程学习|课程学习]]策略**：
- 前 50% 训练：主要采样 Wikipedia、新闻
- 中间 30%：增加书籍、学术论文
- 最后 20%：采样代码、数学文本

**效果**：相比随机采样，在 MMLU 基准上提升 3.2%。

### GPT-4 预训练

**数据**：约 13 万亿 token（推测）

**数据选择**：
- 用分类器过滤低质网页
- 用 RL 优化不同来源的采样比例
- 人工审核高风险数据（医疗、法律）

**去重**：
- MinHash 去重，阈值 0.8
- RL 优化阈值，平衡去重和信息保留

### DeepSeek-V3 预训练

**数据**：14.8 万亿 token

**创新**：
- Multi-head Latent Attention (MLA) 提高效率
- Mixture of Experts (MoE) 扩展容量
- [[01-原子/课程学习|课程学习]]：从简单到复杂

---

## 与下一阶段的关系

**Pre-Training → Mid-Training**：

```
Base Model
  ↓
领域数据继续预训练（数学、代码、推理）
  ↓
+ RL 优化（自我对弈、推理 RL）
  ↓
Capable Model（有特定能力）
```

**关键转变**：
- Pre-Training：学通用知识，RL 是辅助
- Mid-Training：学特定能力，RL 是核心驱动力

---

## 总结

**Pre-Training 阶段的 RL 应用**：

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| 课程学习 | 数据难度不均 | 收敛速度 +20-30% |
| 数据选择 RL | 数据质量参差 | 性能 +10-15% |
| 数据去重 RL | 数据重复 | 减少过拟合 |

**核心洞察**：
- Pre-Training 主要还是自监督学习
- RL 用于优化数据策略，提高效率
- 这是 RL 在 LLM 训练中渗透最浅的阶段
- 未来的趋势是 RL 会更多地参与预训练

---

## 2026 前沿发现（基于深度调研）

> 以下内容来自 2024-2026 前沿论文的深度调研，详见 [[06-LLM训练全流程/深度调研报告-LLM训练中的RL技术]]。

### 发现：RL 从预训练早期就有效

**哈佛 2026.06 研究**（arXiv 2606.04272）发现：RL（RLVR with [[02-模块/Policy-Based/GRPO|GRPO]]）从预训练仅 4B token 的检查点就有效，在 1B 模型上将 GSM8K pass@1 从 ~2% 提升到 ~18%。

**关键洞察：数据组成 > 模型规模**

对于更难的问题（MATH 风格），针对性的预训练数据组成比扩大模型规模更有效：
- 添加 10B 数学专用 token 的 RL 增益 > 同等 token 预算下从 1B 扩展到 4B 参数
- 到 10B token 时，直接 RL 匹配完整的 SFT-then-RL 流水线

**注意**：仅在 1B/4B 规模验证，70B+ 外推未证实。

### 发现：直接 RL 扩展分布，SFT 后才 RL 会收窄

**反直觉发现**：直接在基础预训练检查点上应用 RL 会**扩展**模型输出分布（pass@1 和 pass@k 都提升）。

而先前报告的"分布收窄效应"（pass@1 上升，pass@k 下降）**仅在 RL 跟随 SFT 之后时出现**——是 SFT 限制了探索，不是 RL 本身。

**机制**："没有预先接触标准答案，模型通过 on-policy 学习探索并发现新的推理路径。"

**注意**：在更难的 MATH 问题上，直接 RL 不如 SFT-then-RL。

**来源**：arXiv 2606.04272 Section 4.1

---

📖 **下一阶段**：[[06-LLM训练全流程/02-Mid-Training阶段]]
