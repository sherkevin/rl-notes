---
aliases: [ORPO, Odds Ratio Preference Optimization]
tags:
  - llm
  - post-training
  - preference-optimization
  - single-stage
created: 2026-06-23
---

# ORPO（Odds Ratio 偏好优化）

> SFT 和对齐一步完成，不需要分阶段。用 odds ratio 替代对数概率差。

---

## 一句话定义

**ORPO = SFT + DPO 合并成一步**，用 odds ratio $\frac{\pi(y|x)}{1 - \pi(y|x)}$ 来衡量偏好。

---

## 动机：两阶段太麻烦

标准流程：
1. **SFT 阶段**：$\mathcal{L}_{\text{SFT}} = -\log \pi(y|x)$
2. **DPO 阶段**：$\mathcal{L}_{\text{DPO}} = -\log \sigma(\cdots)$

**问题**：
- 需要训练两次
- SFT 的模型可能不是最好的起点
- 两个阶段的目标不一致

**ORPO 的解决**：一步完成。

---

## Odds Ratio

**定义**：事件发生的概率与不发生的概率之比。

$$\text{odds}(y|x) = \frac{\pi(y|x)}{1 - \pi(y|x)}$$

**对数 odds ratio**：

$$\log \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)} = \log \frac{\pi(y_w|x)}{1 - \pi(y_w|x)} - \log \frac{\pi(y_l|x)}{1 - \pi(y_l|x)}$$

---

## ORPO 的损失函数

$$\mathcal{L}_{\text{ORPO}} = \underbrace{\mathbb{E}_{(x, y_w)} [-\log \pi_\theta(y_w|x)]}_{\text{SFT 项}} + \lambda \cdot \underbrace{\mathbb{E}_{(x, y_w, y_l)} \left[ -\log \sigma \left( \log \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)} \right) \right]}_{\text{偏好项}}$$

**直观理解**：
- **SFT 项**：让模型学会生成好回答
- **偏好项**：让好回答的 odds 比坏回答高

---

## 与 DPO 的对比

| | DPO | ORPO |
|---|---|---|
| 训练阶段 | 2 阶段（SFT + DPO） | 1 阶段 |
| 需要 reference model | ✅ | ❌ |
| 损失函数 | 纯偏好 | SFT + 偏好 |
| 起点 | SFT 模型 | 预训练模型 |
| 效果 | 好 | 相当 |

---

## 优缺点

### 优点

- ✅ 一步完成（省时间）
- ✅ 不需要 reference model
- ✅ 不需要先做 SFT

### 缺点

- ❌ 超参数 $\lambda$ 需要调
- ❌ 仍然是 offline 方法
- ❌ 对数据质量敏感

---

## 适用场景

- 快速原型
- 不想分阶段训练
- 有高质量的偏好数据

---

## 参考资源

- 论文：Hong et al. "ORPO: Monolithic Preference Optimization without Reference Model" (2024)
- 代码：[xfactlab/orpo](https://github.com/xfactlab/orpo)
