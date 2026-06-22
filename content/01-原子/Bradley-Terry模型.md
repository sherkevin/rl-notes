---
tags:
  - atomic
  - preference-modeling
  - rlhf
created: 2026-06-22
---

# Bradley-Terry 模型

> 偏好建模的理论基础：将成对比较转化为概率排序。RLHF 中用于训练 reward model。

---

## 核心思想

给定两个 response $y_w$（赢）和 $y_l$（输），Bradley-Terry 模型假设：

**人类选择 $y_w$ 的概率取决于两者的奖励分数差**：

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中：
- $r_\phi(x, y)$ 是 reward model 对 (prompt, response) 对的打分
- $\sigma$ 是 sigmoid 函数，将分数差映射到概率 [0, 1]

**直觉**：
- 如果 $r(y_w) \gg r(y_l)$，则 $P \to 1$（几乎肯定选 $y_w$）
- 如果 $r(y_w) \approx r(y_l)$，则 $P \approx 0.5$（难以判断）
- 如果 $r(y_w) \ll r(y_l)$，则 $P \to 0$（几乎不会选 $y_w$）

---

## 训练目标

给定标注数据 $\{(x, y_w, y_l)\}$，用最大似然估计训练 reward model：

$$L(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**直觉**：让赢的 response 得分尽量高于输的，差距越大越好。

---

## 为什么用概率而不是直接比较

如果直接用 $r(y_w) > r(y_l)$ 作为损失函数：
- 梯度在 $r(y_w) = r(y_l)$ 处不连续
- 无法表达"稍微好一点"vs"好很多"的程度差异
- 训练不稳定

用 sigmoid 概率：
- 处处可导，梯度平滑
- 自然表达偏好强度
- 与人类偏好的不确定性一致

---

## 局限性

- **只学到序（ranking），没学到绝对值**
  - 两个 response 差 0.1 分和差 10 分，模型不知道
  - 只知道 $r(y_w) > r(y_l)$，不知道大多少

- **需要成对比较数据**
  - 标注成本高，每个偏好对都需要人工比较
  - 无法利用单条标注数据

---

## 被以下模块使用

- [[02-模块/Reward-Model/Pairwise-RM]]：直接应用 Bradley-Terry 训练
- [[02-模块/Reward-Model/DPO]]：隐式 reward 基于 Bradley-Terry 推导

---

## 延伸

- **Plackett-Luce 模型**：Bradley-Terry 的多选项扩展，用于排序多个 response
- **Thurstone 模型**：假设奖励分数服从正态分布，用 CDF 替代 sigmoid
