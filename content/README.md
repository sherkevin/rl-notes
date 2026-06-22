# Reinforcement Learning 复习入口

先看这一份：

1. [[强化学习统一复习地图]]：唯一主入口，合并全景、分类、演化、公式和场景选择。

需要展开时再看：

2. [[00-基础概念/基础概念]]：MDP、回报、价值函数等地基。
3. [[00-基础概念/Bellman方程和TD误差]]：公式体系的核心。
4. [[01-分类维度/Value-Based-Methods]]、[[01-分类维度/Policy-Based-Methods]]、[[01-分类维度/Actor-Critic-Methods]]：需要按分类展开时再看。

最短复习顺序：

`统一复习地图 -> Bellman -> DQN 系 -> PPO/SAC 系 -> Offline/RLHF 系`

深度专题（04-演化综述/）：

5. [[04-演化综述/Value-Based演化史]]：从 Bellman 到 Diffusion-RL，18 个方法的完整演化链。
6. [[04-演化综述/Policy-AC演化史]]：从 REINFORCE 到 DAPO，14 个方法 + 三条演化线（通用策略优化 / 连续控制 / LLM 对齐）。
7. [[04-演化综述/Model-Based演化史]]：从 DP 到 Genie/GameNGen，21 个方法 + 三条演化线（动力学模型 / 世界模型 / 搜索规划）。
8. [[04-演化综述/Offline-RL与RLHF演化史]]：从 BC 到 SimPO，21+ 个方法 + 两大主线（离线数据利用 / 偏好优化）。
9. [[04-演化综述/MARL全景综述]]：从博弈论到 CICERO，涵盖值分解、策略梯度、通信、自我博弈等 6 大板块。
