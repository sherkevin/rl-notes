---
aliases: [KTO, Kahneman-Tversky Optimization]
tags:
  - llm
  - post-training
  - preference-optimization
  - binary-feedback
created: 2026-06-23
---

# KTO（Kahneman-Tversky 优化）

> 不需要成对偏好数据，只需要二值反馈（好/坏）。基于前景理论。

---

## 一句话定义

**KTO = 用 Kahneman-Tversky 前景理论替代 Bradley-Terry 模型**，只需要"这个回答好/坏"的标注，不需要"这个回答比那个好"。

---

## 动机：偏好数据太贵

[[02-模块/Reward-Model/DPO|DPO]] 需要成对的偏好数据 $(y_w, y_l)$：

- 标注者需要比较两个回答
- 成本高、一致性差
- 很多时候只有"这个回答好/坏"的反馈

**KTO 的问题**：能不能只用二值反馈 $(y, \text{good/bad})$？

---

## 理论基础：前景理论

Kahneman-Tversky 前景理论（1979）：

> 人们对损失比收益更敏感（loss aversion）。

$$v(x) = \begin{cases} x^\alpha & \text{if } x \geq 0 \text{ (gain)} \\ -\lambda (-x)^\beta & \text{if } x < 0 \text{ (loss)} \end{cases}$$

其中 $\lambda > 1$（损失厌恶系数，通常 2.25）。

---

## KTO 的损失函数

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x, y)} \left[ \max \left( 0, 1 - v \left( \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0 \right) \right) \right]$$

其中：
- $v(\cdot)$ 是前景理论的价值函数
- $z_0$ 是参考点（通常设为 0）
- 对于好回答：$\frac{\pi_\theta}{\pi_{\text{ref}}} > 1$，$\log$ 为正（gain）
- 对于坏回答：$\frac{\pi_\theta}{\pi_{\text{ref}}} < 1$，$\log$ 为负（loss）

**直观理解**：
- 好回答：让 $\pi_\theta(y|x) > \pi_{\text{ref}}(y|x)$（gain）
- 坏回答：让 $\pi_\theta(y|x) < \pi_{\text{ref}}(y|x)$（loss，且更敏感）

---

## 与 DPO 的对比

| | DPO | KTO |
|---|---|---|
| 数据格式 | $(x, y_w, y_l)$ 成对 | $(x, y, \text{good/bad})$ 单条 |
| 标注成本 | 高（比较两个回答） | 低（判断好坏） |
| 理论依据 | Bradley-Terry | Kahneman-Tversky |
| 损失厌恶 | ❌ | ✅（$\lambda > 1$） |
| 效果 | 好 | 相当 |

---

## 优缺点

### 优点

- ✅ 数据标注简单（二值反馈）
- ✅ 可以利用更多数据（不需要成对）
- ✅ 损失厌恶有助于避免坏回答

### 缺点

- ❌ 仍然需要 reference model
- ❌ 信息量不如成对偏好
- ❌ 超参数多（$\alpha, \beta, \lambda$）

---

## 适用场景

- 只有二值反馈（点赞/点踩）
- 成对偏好数据稀缺
- 需要利用大量单条标注数据

---

## 参考资源

- 论文：Ethayarajh et al. "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)
- 代码：[contextual-ai/HALOs](https://github.com/contextual-ai/HALOs)
