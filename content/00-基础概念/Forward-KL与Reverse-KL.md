---
tags:
  - foundational
  - kl-divergence
  - information-theory
aliases:
  - KL散度
  - Kullback-Leibler Divergence
created: 2026-06-22
---

# Forward KL 与 Reverse KL

> **一句话**：Forward KL 说"P 有的你必须有"（覆盖）；Reverse KL 说"你想有的 P 也得有"（约束）。RL 选 Reverse KL 是因为我们希望新策略果断聚焦，而不是模糊覆盖。

---

## 1. KL 散度到底在度量什么

KL 散度度量的是：**用分布 Q 来"代替"分布 P 时，要多付多少信息代价。**

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right]$$

### 逐步展开（三步）

**第一步：对数除法变减法**

利用对数基本性质 $\log \frac{a}{b} = \log a - \log b$：

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}[\log P(x) - \log Q(x)]$$

**第二步：期望的线性性，拆成两项**

$$\mathbb{E}[A - B] = \mathbb{E}[A] - \mathbb{E}[B]$$

所以：

$$= \mathbb{E}_{x \sim P}[\log P(x)] - \mathbb{E}_{x \sim P}[\log Q(x)]$$

**第三步：认出第一项就是 $-H(P)$**

熵的定义是 $H(P) = -\mathbb{E}_{x \sim P}[\log P(x)]$，所以第一项 $\mathbb{E}_{x \sim P}[\log P(x)]$ 就是 $-H(P)$：

$$= \underbrace{-H(P)}_{\text{P 的熵，常数}} - \underbrace{\mathbb{E}_{x \sim P}[\log Q(x)]}_{\text{交叉熵，真正驱动优化的项}}$$

### 为什么 $-H(P)$ 是常数

它只跟 P 有关，跟你要优化的 Q 完全无关。P 是固定的真实分布/数据分布，你改不了它，所以优化 $D_{KL}$ 时这一项不影响梯度方向，可以丢掉。

最终：**最小化 KL = 最小化第二项（交叉熵）**。这就是为什么交叉熵 loss 在数学上等价于最小化 Forward KL。

---

## 2. 关键直觉：谁采样，谁就有"话语权"

期望 $\mathbb{E}_{x \sim P}$ 意味着**只有 P 认为"重要的地方"才会参与计算**。

- P 概率为 0 的地方 → 不会出现在采样中 → Q 在那里做什么完全无所谓
- P(x) > 0 但 Q(x) → 0 → $\log Q(x) \to -\infty$ → loss 爆炸 → **Q 在 P 有概率的地方绝对不能给零**

这就是 Forward KL 的核心行为。

---

## 3. 一个例子讲透：双峰分布

假设真实分布 P 是**双峰分布**（两个山包），你要用一个**单峰高斯 Q** 去近似它。

```
P (真实，双峰):          Q 的选择空间:

    ╱╲     ╱╲               ╱╲
   ╱  ╲   ╱  ╲     ←→     ╱  ╲    (只有一个峰)
──╱────╲─╱────╲──       ─╱────╲─
  ─────────────            ──────
```

### Forward KL $D_{KL}(P \| Q)$ → 零避免 / 均值寻求

期望基于 P 采样，P 的两个峰都会被采到：

- Q 只盖住左边 → 右边峰上的样本让 $\log Q(x) \to -\infty$ → loss 爆炸
- Q 只盖住右边 → 同理爆炸
- **唯一解：Q 放在两个峰的中间，宽而扁地覆盖两个峰**

```
Forward KL 的结果:

    ╱╲     ╱╲
   ╱  ╲   ╱  ╲    P (双峰)
──╱────╲─╱────╲──
  ╱────────────╲
 ╱      Q       ╲   ← Q 放在中间，宽而扁，覆盖两个峰
```

Q 宁可"模糊"（方差大），也不漏掉 P 的任何部分。

### Reverse KL $D_{KL}(Q \| P)$ → 零强制 / 模式寻求

期望基于 Q 采样，**只有 Q 认为重要的地方才参与计算**：

- Q 盖住左边峰 → 采到的样本都在左边 → P 在那里也高 → $\log P(x)$ 正常 → loss 不大
- Q 在右边峰的概率是 0 → 不会从右边采样 → **右边峰完全不影响 loss**
- **解：Q 精确贴合其中一个峰，忽略另一个**

```
Reverse KL 的结果:

    ╱╲     ╱╲
   ╱  ╲   ╱  ╲    P (双峰)
──╱────╲─╱────╲──
  ╱╲
 ╱  ╲              ← Q 精确贴合一个峰，不管另一个
```

Q 宁可漏掉 P 的一部分，也要在自己覆盖的地方精确。

### 对比表

| | Forward KL $D(P\|Q)$ | Reverse KL $D(Q\|P)$ |
|---|---|---|
| **采样分布** | $x \sim P$ | $x \sim Q$ |
| P 有概率但 Q 给零 | 💥 loss 爆炸 | ✅ 无所谓（采不到）|
| Q 有概率但 P 给零 | ✅ 无所谓 | 💥 loss 爆炸 |
| **行为** | Q 覆盖 P 全部（宁可模糊）| Q 精确贴合 P 的一部分（宁可漏）|
| **别名** | Zero-Avoiding / Mean-Seeking | Zero-Forcing / Mode-Seeking |

---

## 4. Forward KL = 交叉熵 = 监督学习

从 Forward KL 出发推导：

$$D_{KL}(P_{data} \| P_{model}) = \underbrace{-H(P_{data})}_{\text{常数}} + \underbrace{H(P_{data}, P_{model})}_{\text{交叉熵}}$$

最小化 Forward KL = 最小化交叉熵 = **最大似然估计 (MLE)**。

$$\theta^* = \arg\min_\theta -\mathbb{E}_{x \sim P_{data}}[\log P_{model}(x)]$$

在分类任务中，这就是 **交叉熵损失 (Cross-Entropy Loss)**，也等价于 **负对数似然 (NLL)**。

**直觉**：在训练数据出现过的地方，模型的预测概率必须高。

### 为什么是 Zero-Avoiding

当 $P_{data}(x) > 0$ 但 $P_{model}(x) \to 0$ 时，$-\log P_{model}(x) \to +\infty$，loss 爆炸。这迫使模型在数据分布有概率的所有地方都分配正概率——**宁可"模糊"也不漏**。

---

## 5. Reverse KL = 强化学习的约束

在 RL 中，$P = \pi_{old}$（旧策略），$Q = \pi_\theta$（新策略）。

### 为什么 RL 选 Reverse KL

**Forward KL $D_{KL}(\pi_{old} \| \pi_\theta)$**：

- 期望基于 $\pi_{old}$ 采样
- 在 $\pi_{old}$ 认为有可能的**所有**动作上，$\pi_\theta$ 必须给正概率
- 如果 $\pi_{old}$ 对所有动作都有微小概率（如 $\epsilon$-greedy），$\pi_\theta$ 也必须覆盖所有动作
- **结果：新策略被"拖宽"，不敢集中** → 学出来的策略模糊、不够果断

**Reverse KL $D_{KL}(\pi_\theta \| \pi_{old})$**：

- 期望基于 $\pi_\theta$ 采样
- 在 $\pi_\theta$ 想提高概率的动作上，$\pi_{old}$ 也必须有一定概率
- 如果 $\pi_\theta$ 想把概率集中到某个好动作上，只要 $\pi_{old}$ 在那里不是零就行
- **结果：新策略可以"收窄"聚焦到最好的动作** → 策略果断、精确

### TRPO/PPO 中的使用

TRPO 直接约束 Reverse KL：

$$\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \hat{A}(s,a)\right] \quad \text{s.t.} \quad D_{KL}(\pi_\theta \| \pi_{old}) \leq \delta$$

PPO 用 clip 替代显式 KL 约束，但思想一致：限制新旧策略的偏离程度。

### Reverse KL 的坑

因为 Q 只在 P 有概率的地方才安全，所以 **Reverse KL 不允许新策略探索旧策略完全没有试过的方向**。如果 $\pi_{old}$ 完全没给某个好动作分配概率，$\pi_\theta$ 也不能去尝试——策略更新被旧策略的"视野"限制住了。

TRPO 的信赖域 $\delta$ 控制了"每次可以偏离多远"，但方向仍然受旧策略约束。

---

## 6. 重要性采样转换（RL 中实际怎么算）

Reverse KL 需要从 Q（新策略）采样，但训练时只有 P（旧策略）的数据。用**重要性采样**权重 $r(x) = Q(x)/P(x)$ 来修正：

$$\mathbb{E}_{x \sim Q}[\log Q(x) - \log P(x)] = \mathbb{E}_{x \sim P}\left[ \frac{Q(x)}{P(x)} \cdot (\log Q(x) - \log P(x)) \right]$$

这个 $r(x) = Q(x)/P(x)$ 就是 PPO 里的 **probability ratio** $\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$。

PPO 的完整目标函数：

$$L(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{old})$$

其中 KL 惩罚项 $\beta \cdot D_{KL}$ 在实践中通常作为自适应正则化项加入。

---

## 7. 总结

```
                  KL 散度的两种用法
                        │
          ┌─────────────┴─────────────┐
          │                           │
   Forward KL D(P‖Q)          Reverse KL D(Q‖P)
   "P 有的你必须有"            "你想有的 P 也得有"
          │                           │
   采样: x ~ P                 采样: x ~ Q
   Zero-Avoiding               Zero-Forcing
   Mean-Seeking                Mode-Seeking
          │                           │
   监督学习 (SFT)              强化学习 (PPO/TRPO)
   交叉熵 / MLE                KL 约束 / 信赖域
   Q 覆盖 P 全部               Q 精确贴合 P 的一部分
   宁可模糊不漏                宁可漏掉要精确
```

## 8. 延伸

- **Jensen-Shannon 散度 (JSD)**：$\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$，其中 $M = \frac{1}{2}(P+Q)$。Forward 和 Reverse KL 的对称折中，GAN 用这个作为判别器的训练目标。
- [[02-具体算法/Policy-Based/PPO]]：PPO 的 clip 机制本质上是 Reverse KL 约束的简化版。
- [[02-具体算法/Policy-Based/TRPO]]：TRPO 是显式约束 Reverse KL 的原版算法。
- [[04-演化综述/Policy-AC演化史]]：从 TRPO 到 PPO 到 GRPO 的演化，KL 约束的形式在不断变化。
