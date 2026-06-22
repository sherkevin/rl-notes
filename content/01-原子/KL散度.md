---
tags:
  - atomic
  - kl-divergence
  - information-theory
aliases:
  - KL散度
  - Kullback-Leibler Divergence
created: 2026-06-22
---

# KL 散度

> 衡量用分布 Q 近似分布 P 时的信息损失。RL 中用于约束策略更新，监督学习中等价于交叉熵。

---

## 定义

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right]$$

### 熵的概念

$-\log P(x)$ 是单个事件的"惊讶度"，$H(P) = -\mathbb{E}_{x \sim P}[\log P(x)]$ 是平均惊讶度。

- $P(x) = 0.99$ → 不惊讶
- $P(x) = 0.01$ → 很惊讶

用 log 是因为独立事件的惊讶度可以相加：$-\log(0.5 \times 0.5) = -\log 0.5 - \log 0.5$

### KL 散度展开

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}[\log P(x)] - \mathbb{E}_{x \sim P}[\log Q(x)]$$

$$= \underbrace{-H(P)}_{\text{常数}} - \underbrace{\mathbb{E}_{x \sim P}[\log Q(x)]}_{\text{交叉熵}}$$

**关键**：最小化 KL = 最小化交叉熵（因为 $H(P)$ 是常数）。

---

## Forward KL vs Reverse KL

### 核心区别：谁采样

| | Forward KL $D(P\|Q)$ | Reverse KL $D(Q\|P)$ |
|---|---|---|
| **采样分布** | $x \sim P$ | $x \sim Q$ |
| P 有概率但 Q 给零 | 💥 loss 爆炸 | ✅ 无所谓 |
| Q 有概率但 P 给零 | ✅ 无所谓 | 💥 loss 爆炸 |
| **行为** | Q 覆盖 P 全部 | Q 精确贴合 P 的一部分 |
| **别名** | Zero-Avoiding / Mean-Seeking | Zero-Forcing / Mode-Seeking |

### 双峰分布的直觉

**Forward KL**：Q 放在两个峰中间，宽而扁地覆盖
```
    ╱╲     ╱╲
   ╱  ╲   ╱  ╲    P
  ╱────────────╲
 ╱      Q       ╲   ← 宁可模糊不漏
```

**Reverse KL**：Q 精确贴合其中一个峰
```
    ╱╲     ╱╲
   ╱  ╲   ╱  ╲    P
  ╱╲
 ╱  ╲              ← 宁可漏掉要精确
```

### 为什么期望只在 P 的非零处有贡献

期望是加权平均，权重是概率：

$$\mathbb{E}_{x \sim P}[f(x)] = \sum_x P(x) \cdot f(x)$$

当 $P(x) = 0$ 时，$P(x) \cdot f(x) = 0$，不管 $f(x)$ 多大，贡献都是零。

**例子**：
| $x$ | $P(x)$ | $Q(x)$ | $P(x) \cdot \log \frac{P(x)}{Q(x)}$ |
|---|---|---|---|
| $a$ | 0.7 | 0.5 | $+0.236$ |
| $b$ | 0.3 | 0.5 | $-0.153$ |
| $c$ | **0** | 0.9 | **0** |

点 $c$：$P(c) = 0$，Q 给什么值都无所谓，贡献为零。

---

## 应用场景

### 监督学习（Forward KL）

$$D_{KL}(P_{data} \| P_{model}) = -H(P_{data}) + H(P_{data}, P_{model})$$

最小化 Forward KL = 最小化交叉熵 = 最大似然估计 (MLE)

$$\theta^* = \arg\min_\theta -\mathbb{E}_{x \sim P_{data}}[\log P_{model}(x)]$$

**直觉**：数据出现的地方，模型预测概率必须高。

### 强化学习（Reverse KL）

$P = \pi_{old}$，$Q = \pi_\theta$

**为什么选 Reverse KL**：
- Forward KL 会让新策略被"拖宽"，不敢集中
- Reverse KL 让新策略可以"收窄"聚焦到最好的动作

**TRPO/PPO 的约束**：

$$\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \hat{A}(s,a)\right] \quad \text{s.t.} \quad D_{KL}(\pi_\theta \| \pi_{old}) \leq \delta$$

**重要性采样**：Reverse KL 需要从 Q 采样，但只有 P 的数据，用 $r(x) = Q(x)/P(x)$ 修正：

$$\mathbb{E}_{x \sim Q}[\log Q(x) - \log P(x)] = \mathbb{E}_{x \sim P}\left[ \frac{Q(x)}{P(x)} \cdot (\log Q(x) - \log P(x)) \right]$$

### 知识蒸馏

**Forward KL 蒸馏**：student 覆盖 teacher 全部模式（模糊但全面）

**Reverse KL 蒸馏**：student 只学 teacher 的一部分模式（锐利但片面）

详见 [[02-模块/Reward-Model/知识蒸馏]]

---

## 被以下模块使用

- [[02-模块/Policy-Based/PPO]]：Reverse KL 约束的简化版（用 clip 替代）
- [[02-模块/Policy-Based/TRPO]]：显式约束 Reverse KL
- [[02-模块/Reward-Model/DPO]]：隐式 reward 基于 KL 推导
- [[02-模块/Reward-Model/知识蒸馏]]：Forward/Reverse KL 的不同效果

---

## 延伸

- **Jensen-Shannon 散度 (JSD)**：$\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$，其中 $M = \frac{1}{2}(P+Q)$。Forward 和 Reverse KL 的对称折中。
