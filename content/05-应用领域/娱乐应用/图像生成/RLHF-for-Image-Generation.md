---
aliases: [RLHF for Image Generation, 图像生成RLHF, 审美对齐]
tags:
  - diffusion
  - rlhf
  - image-generation
  - aesthetic
created: 2026-06-24
---

# RLHF for Image Generation

> 用人类反馈优化扩散模型，让生成的图像更美、更符合意图。

---

## 一句话定义

**RLHF for Image Generation = Diffusion 模型 + 人类审美偏好 + 奖励模型 + RL 优化**，让 AI 生成的图像从"技术上正确"变成"审美上优秀"。

---

## 为什么需要 RLHF？

### Diffusion 模型的问题

预训练的 Diffusion 模型（如 Stable Diffusion）能生成技术上正确的图像，但：

- **审美不一致**：有时生成很美的图，有时很丑
- **文本-图像对齐差**：prompt 说"美丽的日落"，生成的可能是普通的日落
- **缺乏人类偏好**：不知道人类喜欢什么样的构图、色彩、风格

### RLHF 的解决

用人类反馈教会模型"什么是美"：

1. **收集偏好数据**：让人类比较两张图哪张更好
2. **训练奖励模型**：学习人类的审美偏好
3. **RL 优化**：用奖励信号微调 Diffusion 模型

---

## 技术流程

### 阶段 1：预训练 Diffusion 模型

标准 Diffusion 训练：

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**结果**：得到基础生成模型 $\pi_{\text{pretrain}}$。

---

### 阶段 2：收集人类偏好数据

**数据格式**：

```
Prompt: "a beautiful sunset over the ocean"

Image A: [生成的图1]  ← 人类标注：更喜欢
Image B: [生成的图2]
```

**标注方式**：

- **Pairwise Comparison**：比较两张图哪张更好
- **Rating**：给每张图打分（1-5 分）
- **Ranking**：对多张图排序

**评估维度**：

- **审美质量**：构图、色彩、光影
- **文本对齐**：是否符合 prompt
- **真实感**：是否像真实照片/画作
- **创意性**：是否有新意

---

### 阶段 3：训练奖励模型

**模型架构**：

- 输入：图像 + prompt
- 输出：标量奖励分数

$$r_\phi(x, y) = \text{RewardModel}(\text{image}, \text{prompt})$$

**训练目标**（Bradley-Terry 模型）：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

其中 $y_w$ 是人类偏好的图像，$y_l$ 是不偏好的。

**现有奖励模型**：

- **HPS v2**（Human Preference Score）：大规模人类偏好训练的通用审美模型
- **ImageReward**：专门针对文本-图像生成的奖励模型
- **PickScore**：基于 Pick-a-Pic 数据集训练

---

### 阶段 4：RL 优化 Diffusion 模型

**挑战**：Diffusion 模型的生成过程是迭代的（50-1000 步），如何定义 RL 的 action 和 reward？

**方法 1：Reward at the End**

只在生成完成后给奖励：

$$R = r_\phi(x_T, \text{prompt})$$

用 REINFORCE 或 PPO 优化：

$$\max_\theta \mathbb{E}_{x_T \sim \pi_\theta}[r_\phi(x_T, \text{prompt})] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{pretrain}})$$

**问题**：奖励稀疏（只在最后一步），梯度方差大。

**方法 2：Reward at Each Step**

在每一步去噪后都给奖励：

$$R_t = r_\phi(x_t, \text{prompt})$$

用 TD 学习优化：

$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_\phi(x_t, \text{prompt}) \right]$$

**优点**：奖励密集，学习更稳定。

**问题**：如何定义中间步骤的"好坏"？

**方法 3：Direct Preference Optimization (DPO)**

跳过奖励模型，直接用偏好数据优化：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**优点**：不需要训练奖励模型，更简单。

**缺点**：Diffusion 模型的对数概率计算复杂。

---

## 代表性工作

### HPS v2（Human Preference Score v2）

**论文**：Lee et al. "Human Preference Score v2" (2023)

**数据**：80 万张图像的人类偏好标注

**模型**：基于 CLIP 的奖励模型

**应用**：评估和优化文本-图像生成模型

**局限**：偏向"流行审美"，可能缺乏多样性

---

### ImageReward

**论文**：Xu et al. "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation" (2023)

**数据**：13 万张图像 + prompt + 人类偏好

**模型**：基于 BLIP 的奖励模型

**特点**：
- 同时考虑审美质量和文本对齐
- 可以用于 RL 优化

**应用**：
- 微调 Stable Diffusion
- 评估不同生成模型

---

### DDPO（Denoising Diffusion Policy Optimization）

**论文**：Black et al. "Training Diffusion Models with Reinforcement Learning" (2023)

**方法**：
- 把 Diffusion 过程建模为 MDP
- 每步去噪是一个 action
- 用 PPO 优化

**结果**：
- 在 HPS 和 ImageReward 上显著提升
- 生成的图像更美、更符合 prompt

---

## 核心挑战

### 1. 奖励模型不准

**问题**：
- 审美是主观的，标注一致性差
- 奖励模型可能过拟合到"流行审美"
- 不同文化、年龄、性别的审美差异大

**解决**：
- 多样化标注者
- 多维度奖励（审美、对齐、真实感分别打分）
- 个性化奖励模型

### 2. Reward Hacking

**问题**：
- 模型找到奖励模型的漏洞
- 生成的图像"看起来很美"但不符合 prompt
- 例如：所有图都加滤镜，因为奖励模型喜欢高对比度

**解决**：
- KL 约束（不要偏离预训练模型太远）
- 多维度奖励（不能只优化一个维度）
- 定期更新奖励模型

### 3. 计算成本

**问题**：
- Diffusion 生成慢（50-1000 步）
- RL 需要大量采样
- 奖励模型推理也慢

**解决**：
- 少步数 Diffusion（DPM-Solver、LCM）
- 并行采样
- 用 DPO 替代 RL（不需要奖励模型推理）

---

## 实践建议

### 如果你要优化图像生成模型

1. **先用现有奖励模型评估**：HPS v2、ImageReward
2. **收集自己的偏好数据**：针对你的领域（如动漫、摄影）
3. **训练领域特定的奖励模型**：不要直接用通用模型
4. **用 DPO 微调**：比 RLHF 简单稳定
5. **多维度评估**：不只看奖励分数，还要看多样性、真实感

### 技术选型

- **基础模型**：Stable Diffusion XL、FLUX
- **奖励模型**：HPS v2（通用）、ImageReward（文本对齐）
- **对齐方法**：DPO（简单）、DDPO（效果更好但复杂）
- **训练框架**：diffusers + TRL

---

## 2024-2025 前沿

### 1. 多模态奖励

**现状**：只看图像质量

**未来**：
- 图像 + 文本的整体协调
- 视频生成的奖励（时序一致性）
- 3D 生成的奖励（多角度一致性）

### 2. 个性化审美

**现状**：通用审美模型

**未来**：
- 用户特定的审美偏好
- 动态调整（根据用户反馈）
- 风格迁移（把用户的审美应用到新领域）

### 3. 创意增强

**现状**：优化"正确性"和"美观"

**未来**：
- 鼓励创意（不只是"正确"的图）
- 探索新颖风格（不只是"流行"的图）
- 艺术表达（支持抽象、实验性作品）

---

## 参考资源

- 论文：Lee et al. "Human Preference Score v2" (2023)
- 论文：Xu et al. "ImageReward" (2023)
- 论文：Black et al. "Training Diffusion Models with Reinforcement Learning" (2023, DDPO)
- 代码：[Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- 教程：[RLHF for Image Generation](https://huggingface.co/blog/rlhf-image-generation)
