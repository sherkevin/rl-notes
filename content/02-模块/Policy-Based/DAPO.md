---
tags:
  - #model-free
  - #actor-critic
  - #on-policy
  - #online-rl
  - #continuous
  - #function-approximation
---

### 1. 算法背景与核心动机
在LLM的推理能力（Reasoning）训练中，主流方法通常使用 **[[PPO]]** 或 **[[GRPO]]** (Group Relative Policy Optimization)。然而，研究团队发现在处理长思维链（Long Chain-of-Thought, CoT）时，这些基线方法存在显著缺陷：

- **熵坍塌 (Entropy Collapse)**: 模型在训练初期容易过早收敛到某种单一的输出模式，导致探索能力丧失。
    
- **训练效率低**: 许多采样的样本对梯度的贡献极小（例如问题太简单全对，或太难全错）。
    
- **奖励噪声**: 在长CoT中，稀疏的奖励（只有最终答案对错）难以有效指导中间步骤。

### 2. DAPO 的四大核心技术 (Key Techniques)

DAPO 对传统的策略优化（Policy Optimization）做了四项关键改进：

#### A. Clip-Higher (非对称截断)

- **问题**: 传统的PPO/GRPO使用对称的 `clip` 操作（限制在 $[1-\epsilon, 1+\epsilon]$），这在防止策略更新过大的同时也限制了模型向“更好”方向探索的幅度，容易导致模型为了“安全”而迅速降低输出的多样性（即熵坍塌）。
    
- **DAPO解法**: 采用**解耦的截断机制**，特意**调高了上限截断值**（Upper Clip Range）。
    
    - 这意味着允许模型在“正向优势”（Positive Advantage）的样本上进行更大幅度的更新。
        
    - **效果**: 保持了策略的熵（Entropy），促进了多样性探索，防止模型过早陷入局部最优。
        

#### B. Dynamic Sampling (动态采样)

- **问题**: 在GRPO中，如果你对一个问题采样了一组回答（Group），如果这组回答全对（Reward全为1）或全错（Reward全为0），它们产生的相对优势（Advantage）通常接近于0，对梯度更新几乎没有贡献，但却占据了大量的计算资源。
    
- **DAPO解法**: 引入**动态采样策略**。
    
    - 在训练过程中，实时监控每个Prompt组的准确率。
        
    - **过滤**: 自动剔除那些准确率为 100%（全对）或 0%（全错）的Prompt组。
        
    - **保留**: 专注于那些准确率在 $(0, 1)$ 之间的样本，即模型“似懂非懂”的边界案例。
        
    - **效果**: 显著提高了训练的样本效率（Sample Efficiency）和稳定性。
        

#### C. Token-Level Policy Gradient Loss (Token级[[01-原子/策略梯度|策略梯度]]损失)

- **细节**: 这是一个针对长文本生成的优化。虽然具体数学形式类似标准PG，但DAPO强调在长CoT场景下，需要精确地在Token级别计算和累积梯度，而不是简单地对整个序列做平均处理，这对于捕捉长推理链中的细微逻辑依赖至关重要。
    

#### D. Overlong Reward Shaping (过长惩罚机制)

- **问题**: 在强化学习激发CoT的过程中，模型有时会学会“刷步数”，即生成冗长但无意义的废话来试图“骗取”或延迟奖励判定，或者仅仅是因为探索过度导致发散。
    
- **DAPO解法**: 引入显式的**长度惩罚（Length Penalty）**作为Reward Shaping的一部分。
    
    - 如果在推理正确的前提下，回答过于冗长，会受到轻微的负奖励惩罚。
        
    - **效果**: 引导模型在保持推理能力的同时，输出更加简洁、高效的思维链。
        

---

### 3. 系统架构与实现 (Based on Verl)

DAPO不仅仅是一个算法，还是一个基于 **Verl** (Volcengine RL) 框架构建的完整开源系统。

- **框架**: 建立在PyTorch和Ray之上，支持大规模分布式训练。
    
- **硬件利用**: 针对超长序列（Sequence Length > 32k甚至更高）进行了显存和计算优化，支持在数百张GPU上进行集群训练。
    
- **数据**: 同时也开源了配套的高质量数学数据集（DAPO-Math-17k）。
    

---

### 4. 性能对比

根据论文报告（2025年3月数据）：

- **基座模型**: Qwen2.5-32B
    
- **Benchmark (AIME 2024)**:
    
    - **DAPO**: 50.0 分
        
    - **DeepSeek-R1-Zero-Qwen-32B**: 47.0 分
        
    - **原始 [[02-模块/Policy-Based/GRPO|GRPO]]**: ~30 分
        
- **结论**: DAPO用更少的训练步数（50% steps）达到了比DeepSeek-R1-Zero更好的效果，证明了其改进策略的有效性。
    

---

### 5. 其他同名算法辨析 (以防混淆)

虽然大概率你问的是上述LLM RL算法，但学术界还有一个较早或不同领域的 "DAPO"：

- **Domain-Adaptive Policy Optimization**: 用于迁移学习或机器人领域的自适应策略优化。
    
- **Mobility-Aware DAPO**: 用于边缘计算（Edge Computing），通过DDPG算法优化模型在边缘设备上的分区和卸载。
    

**鉴于你的清华背景和AI Agent兴趣，请主要关注前文描述的 LLM Reasoning DAPO。**

### 我们可以做的下一步

DAPO 的代码和论文都已经开源（GitHub: BytedTsinghua-SIA/DAPO 或类似路径）。

你想让我帮你解析一下它在 clip_higher 部分的具体代码实现逻辑，还是想了解如何用Verl框架复现这个算法？