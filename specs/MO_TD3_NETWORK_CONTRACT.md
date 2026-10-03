# MO-TD3 网络契约

全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。实现：[`../pmo_evorl/networks.py`](../pmo_evorl/networks.py)。

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

## 与 EvoRL 标准 TD3 的差异

EvoRL 标准 Actor 只输入观测，Critic 输出标量 Q。本项目不改动上游网络工厂，而在项目命名空间内增加偏好条件网络；标量化和 Twin Critic 选择由后续 MO-TD3 loss 实现。
