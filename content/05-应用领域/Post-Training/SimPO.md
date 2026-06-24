---
aliases: [SimPO, Simple Preference Optimization]
tags:
  - llm
  - post-training
  - preference-optimization
  - no-reference
created: 2026-06-23
---

# SimPO（简单偏好优化）

> DPO 的改进版，不需要 reference model，更简单高效。

---

## 一句话定义

**SimPO = DPO - reference model + length normalization**，用平均对数概率替代显式的 KL 约束。

---

## 动机：DPO 的问题

[[02-模块/Reward-Model/DPO|DPO]] 需要一个 reference model $\pi_{\text{ref}}$ 来计算 KL 约束：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

**问题**：
1. **需要额外模型**：$\pi_{\text{ref}}$ 占内存
2. **推理时也要跑两个模型**：慢
3. **$\pi_{\text{ref}}$ 可能过时**：训练过程中 $\pi_\theta$ 变化，但 $\pi_{\text{ref}}$ 固定

---

## SimPO 的解决方案

**核心思想**：用**序列的平均对数概率**作为隐式奖励，不需要显式 reference model。

$$r_\theta(x, y) = \frac{\beta}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i | x, y_{<i})$$

其中 $|y|$ 是序列长度，$\beta$ 是温度参数。

**为什么除以长度？**
- 长序列的对数概率之和天然更大
- 除以长度后，奖励与长度无关，更公平

---

## 损失函数

$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

展开：

$$= -\log \sigma \left( \frac{\beta}{|y_w|} \sum_{i} \log \pi_\theta(y_{w,i} | \cdots) - \frac{\beta}{|y_l|} \sum_{j} \log \pi_\theta(y_{l,j} | \cdots) \right)$$

**直观理解**：
- $r_\theta(x, y_w)$ 是"好回答的平均对数概率"
- $r_\theta(x, y_l)$ 是"坏回答的平均对数概率"
- 让好的比坏的高

---

## 与 DPO 的对比

| | DPO | SimPO |
|---|---|---|
| 需要 reference model | ✅ | ❌ |
| 奖励定义 | $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | $\frac{\beta}{|y|} \sum \log \pi_\theta$ |
| 长度偏差 | 有（长序列得分低） | 无（除以长度） |
| 内存占用 | 2 个模型 | 1 个模型 |
| 训练速度 | 较慢 | 较快 |
| 效果 | 好 | 相当或更好 |

---

## 长度归一化的重要性

**问题**：DPO 倾向于生成**短回答**。

原因：
- 长序列的对数概率之和更负（每个 token 的概率 < 1）
- DPO 为了让 $\pi_\theta(y_w|x)$ 大，倾向于选短的 $y_w$

**SimPO 的解决**：除以长度 $|y|$。

$$\text{平均对数概率} = \frac{1}{|y|} \sum_{i} \log \pi_\theta(y_i | \cdots)$$

这样长序列和短序列的奖励可比，模型不会偏向短回答。

---

## 实践建议

### 什么时候用 SimPO？

- 内存受限（不想存 reference model）
- 需要快速迭代
- 数据中有长度差异大的回答

### 超参数

- $\beta$：通常 2.0-2.5（比 DPO 的 0.1 大很多）
- 学习率：与 DPO 类似，1e-6 ~ 5e-6
- Epochs：2-3

---

## 优缺点

### 优点

- ✅ 不需要 reference model（省内存）
- ✅ 训练更快
- ✅ 没有长度偏差
- ✅ 效果与 DPO 相当或更好

### 缺点

- ❌ $\beta$ 需要调参
- ❌ 仍然是 offline 方法，探索能力弱
- ❌ 对数据质量敏感

---

## 演化位置

```
DPO (2023)
    ↓
SimPO (2024) ← **你在这里**
    ↓
KTO (2024)
```

详见 [[05-应用领域/Post-Training/Post-Training方法全景]]

---

## 参考资源

- 论文：Meng et al. "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024)
- 代码：[princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)
