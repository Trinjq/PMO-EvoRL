# MO-TD3 网络契约

全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。网络实现：[`../pmo_evorl/networks.py`](../pmo_evorl/networks.py)。Target 与 loss：[`../pmo_evorl/mo_td3.py`](../pmo_evorl/mo_td3.py)。

## 源码基线

以原 PD-MORL `lib/models/networks.py` 中的 `Actor` 和 `Critic` 为准：

- Actor 连接状态 `s` 与偏好 `w`，经过两个400单元 ReLU 隐层，以 `tanh` 输出6维动作。
- Twin Critic 是参数独立的两个网络；每个连接 `s`、`w` 和 `a`，经过两个400单元 ReLU 隐层，输出2维 Vector Q。
- 线性层使用 Xavier normal 权重初始化和零偏置，与原 PyTorch 实现对齐。

## 批量形状

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| Actor | `state [B,17]`, `preference [B,2]` | `action [B,6]` |
| Twin Vector Critic | `state [B,17]`, `preference [B,2]`, `action [B,6]` | `vector_q [B,2 critics,2 objectives]` |
| 标量化（后续 loss 模块） | `preference [B,2]`, `vector_q [B,2,2]` | `w^TQ [B,2 critics]` |

`w` 直接条件化 Actor/Critic。投影偏好 `wp` 不进入这两个网络，后续只供 directional-angle loss 使用。

## Target 与 loss 契约

- Target Actor 使用 `(s', w)` 生成下一动作，加入裁剪后的 target policy noise，再把动作裁剪到 `[-1,1]`。
- 两个 Target Critic 输出 `[B,2 critics,2 objectives]`；分别用原偏好 `w` 得到标量 `w^TQ`，选取标量较小的 Critic，保留它的整个二维 Q 向量。
- Vector TD Target 为 `r + gamma * (1-done) * selected_vector_q`。原码的 `done` 同时包含环境终止和 Gym `TimeLimit`；迁移版不直接沿用 EvoRL 标准 TD3 “只屏蔽 termination”的行为。
- Critic loss 对每个 Critic 计算 `directional_angle(wp,Q) + SmoothL1(Q,target)` 后求和。
- Actor loss 仅使用 Q1：`-mean(w^TQ1) + 10 * mean(directional_angle(wp,Q1))`。
- Directional angle 按原码把 cosine 裁剪到 `[0,0.9999]` 后计算角度制 `acos`。

## 三个 Key 与 Interpolator

原始 `train_Walker2d_MO_TD3_HER_Key.py` 先分别在 `w=(0,1)`、`(0.5,0.5)`、`(1,0)` 上训练三个 Key 策略，生成 `interp_objs_walker2d.txt`。官方文件中保存的三个目标向量为：

```text
[ 497.2413299560547, 2494.5884033203124]
[1639.5962036132812, 2156.5128173828125]
[2602.103564453125,  691.5010070800781]
```

训练主程序启动时对它们做 L2 归一化，然后构建 `RBFInterpolator(kernel="linear")`。[`../pmo_evorl/interpolator.py`](../pmo_evorl/interpolator.py) 用纯 JAX 的4×4增广线性系统复现 SciPy 的 linear RBF 与常数多项式项，避免训练主路径发生 CPU 回调。

源码在训练期间更新 Key 目标后改用 L1 归一化，与启动时的 L2 不一致。迁移版保留可显式选择的归一化函数，在动态更新尚未接入前使用源码启动行为 L2。

## 与 EvoRL 标准 TD3 的差异

EvoRL 标准 Actor 只输入观测，Critic 输出标量 Q。本项目不改动上游网络工厂，而在项目命名空间内增加偏好条件网络与 MO-TD3 目标函数。
