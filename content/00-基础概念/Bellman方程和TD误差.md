![image.png](https://gitee.com/sherkevin/pictures/raw/master/20250930150430.png)
这个误差值 `δ_t` 可能为正，也可能为负：
- 如果 **TD误差是正数**：说明“现实”比“预期”要好，我们的 `Q(s, a)` 估值得太低了，需要把它**调高**一些。
- 如果 **TD误差是负数**：说明“现实”比“预期”要差，我们的 `Q(s, a)` 估值得太高了，需要把它**调低**一些。
- 如果 **TD误差是0**：说明“现实”和“预期”完全一致，`Q(s, a)` 已经满足了贝尔曼方程，**无需更新**。
这整个方程就是贝尔曼方程思想构建出来的：now_state_value = r + next_state_value
>Temporal-Difference Learning



