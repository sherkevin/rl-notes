这是一个非常前沿且专业的问题。**GRPO (Group Relative Policy Optimization，群体相对策略优化)** 是由 **DeepSeek** 团队（在 DeepSeekMath 和 DeepSeek-R1 等研究中）提出的一种强化学习算法。

它的核心目标是：**在不使用价值函数模型（Critic Model）的情况下，大幅降低大语言模型强化学习（RLHF/RL）的训练成本，同时保持高效的性能。**

下面我将从数学公式出发，深入拆解 GRPO 的细节，并结合图示和例子帮助你理解。

---

# GRPO（Group Relative Policy Optimization）

### 1. 核心背景：为什么要发明 GRPO？

在传统的 **[[02-模块/Policy-Based/PPO|PPO]] (Proximal Policy Optimization)** 算法中，我们通常需要四个模型（或者至少两个主模型）：

1. **Actor (策略模型):** 生成回答。
    
2. **Critic (价值模型):** 评估当前状态（Prompt）或动作（Response）的价值 $V(s)$。
    
3. **Reference Model:** 用于计算 [[01-原子/KL散度|KL 散度]]，防止模型跑偏。
    
4. **[[01-原子/Reward-Model训练方法|Reward Model]]:** 给回答打分。
    

**痛点：** Critic 模型通常需要和 Actor 模型一样大。如果你在训练一个 70B 的模型，[[02-模块/Policy-Based/PPO|PPO]] 需要同时加载 Actor 和 Critic，显存消耗巨大，且训练速度慢。

**GRPO 的解决方案：** **直接去掉 Critic 模型。** 它利用“群体采样（Group Sampling）”生成的多个样本，通过计算它们之间的**相对优劣**来代替 Critic 对价值的预估。

---

# GRPO（Group Relative Policy Optimization）

### 2. GRPO 的数学原理与公式推导

GRPO 的核心公式基于 [[02-模块/Policy-Based/PPO|PPO]]，但在**[[01-原子/优势函数|优势函数]]（Advantage Function）**的计算上做了重大创新。

#### 2.1 目标函数 (Objective Function)

GRPO 的目标是最大化以下目标函数：

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(q)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$

让我们逐项拆解这个复杂的公式：

1. **输入与采样**:
    
    - $q$: 问题（Prompt）。
        
    - $\{o_i\}_{i=1}^G$: 针对同一个问题 $q$，旧策略 $\pi_{\theta_{old}}$ 生成的一组输出（Group）。$G$ 是组的大小（例如 $G=64$）。
        
    - 这意味着：对于每个问题，模型一次性生成 $G$ 个不同的回答。
        
2. **策略比率 (Policy Ratio)**:
    
    - $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$: 这与 [[02-模块/Policy-Based/PPO|PPO]] 一致，衡量新策略 $\pi_\theta$ 相对于旧策略生成该回答的概率变化。
        
3. **剪裁 (Clipping)**:
    
    - $\text{clip}(..., 1-\epsilon, 1+\epsilon)$: 限制更新幅度，防止策略更新过猛导致训练崩溃。这继承自 [[02-模块/Policy-Based/PPO|PPO]]。
        
4. **[[01-原子/KL散度|KL 散度]] (KL Divergence)**:
    
    - $- \beta D_{KL}(\pi_\theta || \pi_{ref})$: 正则化项。确保训练中的模型 $\pi_\theta$ 不会偏离原始的基础模型（SFT模型）$\pi_{ref}$ 太远，防止语言能力崩坏。
        

#### 2.2 核心创新：[[01-原子/优势函数|优势函数]] $A_i$ (Advantage)

在 [[02-模块/Policy-Based/PPO|PPO]] 中，优势 $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，这依赖于 Critic 模型估算的 $V(s)$。

在 **GRPO** 中，优势是通过**组内标准化（Group Normalization）**计算的：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

- $r_i$: 第 $i$ 个回答的原始奖励（由 [[01-原子/Reward-Model训练方法|Reward Model]] 或 规则打分）。
    
- $\text{mean}(...)$: 这组 $G$ 个回答的平均奖励。
    
- $\text{std}(...)$: 这组 $G$ 个回答的标准差。
    

直觉解释：

GRPO 不关心绝对分数是 10 分还是 100 分。它只关心：在这个问题下，回答 $i$ 相比于刚才生成的其他 $G-1$ 个回答，是更好还是更差？

- 如果 $A_i > 0$: 说明这个回答优于平均水平，应该提高其生成概率。
    
- 如果 $A_i < 0$: 说明这个回答劣于平均水平，应该降低其生成概率。
    
- **平均值实际上充当了 Baseline (基线)，从而取代了 Critic 的作用。**
    

---

# GRPO（Group Relative Policy Optimization）

### 3. 算法流程细节

下面是 GRPO 训练一个 Step 的详细流程：

1. **采样 (Sampling):**
    
    - 从数据集采样一批问题 $q$。
        
    - 对于每个 $q$，使用当前模型 $\pi_{\theta_{old}}$ 采样生成 $G$ 个输出 $\{o_1, o_2, ..., o_G\}$。
        
    - _注意：这里不需要 Critic 参与前向传播。_
        
2. **打分 (Evaluation):**
    
    - 使用奖励模型（RM）或基于规则的环境（如数学题答案验证器）计算每个输出的奖励 $\{r_1, r_2, ..., r_G\}$。
        
    - 同时计算参考模型 $\pi_{ref}$ 的概率，用于计算 KL 散度。
        
3. **优势计算 (Advantage Estimation):**
    
    - 对这 $G$ 个奖励进行标准化处理：$A_i = (r_i - \mu) / \sigma$。
        
4. **策略优化 (Policy Optimization):**
    
    - 将 $A_i$ 代入目标函数。
        
    - 计算梯度并更新模型参数 $\theta$，最大化表现好的样本的概率，抑制表现差的样本。
        

---

# GRPO（Group Relative Policy Optimization）

### 4. 具体例子：教模型做数学题

假设我们要训练 DeepSeek-Math 解决一个数学问题。

**输入 (Prompt):** "计算 $2x + 3 = 7$，求 $x$。"

步骤 1：生成组 (Group Generation)

设定组大小 $G=4$。模型 $\pi_{\theta_{old}}$ 生成了以下 4 个推理过程和答案：

- **$o_1$:** "移项得 $2x = 4$，除以 2 得 $x=2$。" (正确，逻辑清晰)
    
- **$o_2$:** "直接猜 $x=2$。" (答案对，但在推理任务中通常分低，或者假设我们只看最终答案) -> 假设我们要求过程，这个给半分。
    
- **$o_3$:** "$2x = 10$，所以 $x=5$。" (计算错误)
    
- **$o_4$:** "不知道。" (拒绝回答)
    

步骤 2：奖励打分 (Reward Calculation)

假设我们有一个根据答案正确性和过程给分的系统：

- $r_1 = 1.0$ (完美)
    
- $r_2 = 0.5$ (对了一半)
    
- $r_3 = 0.0$ (错误)
    
- $r_4 = 0.1$ (至少说了话)
    

**步骤 3：计算优势 (Calculate Advantage)**

- **平均值 (Mean):** $(1.0 + 0.5 + 0.0 + 0.1) / 4 = 0.4$
    
- **标准差 (Std):** 计算这组数的标准差，约等于 $0.39$ (粗略估算)。
    

现在计算每个回答的优势 $A_i$：

- **$A_1$ (针对 $o_1$):** $(1.0 - 0.4) / 0.39 \approx \mathbf{+1.54}$
    
    - _含义：_ 表现极好，大幅**增加**生成该路径的概率。
        
- **$A_2$ (针对 $o_2$):** $(0.5 - 0.4) / 0.39 \approx \mathbf{+0.25}$
    
    - _含义：_ 略好于平均，微幅增加概率。
        
- **$A_3$ (针对 $o_3$):** $(0.0 - 0.4) / 0.39 \approx \mathbf{-1.02}$
    
    - _含义：_ 表现差，大幅**降低**生成该路径的概率。
        

**结论：** 即使没有 Critic 告诉模型“0.4分是平均水平”，模型通过自己跟自己生成的其他结果对比（Peer Review），也知道了 $o_1$ 是它应该学习的方向。

---

# GRPO（Group Relative Policy Optimization）

### 5. GRPO 相比 [[02-模块/Policy-Based/PPO|PPO]] 的优缺点总结

|**特性**|**[[02-模块/Policy-Based/PPO|PPO]] (Proximal Policy Optimization)**|**GRPO (Group Relative Policy Optimization)**|
|---|---|---|
|**模型结构**|需要 Actor 和 Critic (显存占用大)|**只需要 Actor** (显存占用小，类似 SFT)|
|**优势计算**|依赖 Critic 的价值估计 $V(s)$|**依赖组内输出的相对平均值**|
|**适用场景**|通用 RLHF|**特别适合推理、数学、代码** (通常有确定性结果)|
|**计算效率**|较慢 (Critic 推理 + 反向传播)|**较快** (省去了 Critic 的开销)|
|**稳定性**|Critic 如果估值不准，训练容易崩|依赖 Group Size，如果 $G$ 太小，方差会很大|

### 总结

GRPO 是一种**去 Critic 化**的策略优化算法。它巧妙地利用了”群体智慧”——即模型自己生成的多个样本的平均表现作为基准线，从而使得模型能够在不需要额外价值网络的情况下进行自我迭代和进化。这正是 DeepSeek 系列模型能够高效进行推理能力强化的关键技术之一。

---

# GRPO（Group Relative Policy Optimization）

### 扩展阅读

- **GRPO 在 RLHF 演化中的位置** ([[02-模块/Policy-Based/PPO|PPO]] → ReMax → RLOO → GRPO → [[02-模块/Policy-Based/DAPO|DAPO]]): 见 [Offline-RL与RLHF-偏好优化演化全景.md](../../Offline-RL与RLHF-偏好优化演化全景.md#b11-grpo-group-relative-policy-optimization-2024)
- **GRPO 变体**: [[02-模块/Policy-Based/DAPO|DAPO]] (Decoupled Clip + Dynamic Sampling, ByteDance 2025)、VIMPO (2026)、N-GRPO (2026)、AdaGRPO (2026)

**接下来你想了解如何用代码（如 PyTorch 或 TRL 库）来实现一个简化版的 GRPO 训练循环吗？**