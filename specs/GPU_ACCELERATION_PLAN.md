# PD-MORL GPU 加速执行计划

入口索引：[`../AGENTS.md`](../AGENTS.md)。全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。实测结果：[`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

## 目标与验收

连续动作 Walker2d 的 MO-TD3-HER 采用严格基线与速度优先版本双轨推进。加速版在相同原始环境步数下必须满足：

- 三个随机种子的平均 Hypervolume 不低于严格基线的95%；
- 任一随机种子的 Hypervolume 不低于对应基线的90%；
- 从进程启动到 CPU Pareto 指标写出的端到端墙钟时间至少提升1.5倍；
- 环境、奖励、终止条件、偏好输入、HER语义和最终 CPU MuJoCo 评估协议不变。

当前 `formal_10m_20261004` 必须自然完成，不得因加速开发中断或修改。

## 配置边界

- `configs/mo_td3_walker2d.yaml` 是严格源码比例基线，不为加速实验修改默认行为。
- `configs/mo_td3_walker2d_fast.yaml` 是速度优先配置，默认对应 A3。
- `key_update_interval` 表示两次动态 Key 更新之间至少增加的完整 episode 轮数；基线值为1。
- `critic_updates_per_transition` 与 `actor_updates_per_transition` 显式记录学习强度，训练调度始终按原始 transition 计算。
- A4 属于算法加速变体，不得描述为与原源码更新轨迹等价。

## 筛选顺序

所有候选先运行100万原始环境步、seed 1，并使用同一 CPU Pareto 协议评估：

| 候选 | 环境数 | Key episodes | Key 间隔 | Critic/transition | Actor/transition |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 320 | 3 | 1 | 1.0 | 0.1 |
| A1 | 640 | 3 | 1 | 1.0 | 0.1 |
| A2 | 640 | 1 | 1 | 1.0 | 0.1 |
| A3 | 640 | 1 | 5 | 1.0 | 0.1 |
| A4 | 640 | 1 | 5 | 0.5 | 0.05 |

执行顺序固定为 B0、A1、A2、A3、A4。出现 NaN、训练失败或 Hypervolume 低于 B0 的90%即停止该候选。只有 A3 的 Key 评估仍超过训练墙钟时间20%时，才考虑 CPU 在线 Key 评估。

## 正式验证

1. 最快且满足质量门槛的两个候选运行300万步、seeds 1/2。
2. 平均 Hypervolume 不低于基线95%且无单 seed 低于90%的最快候选晋级。
3. 最终候选运行1000万步、seeds 1/2/3；每张可用 GPU 运行一个独立 seed。
4. 所有最终 checkpoint 统一执行 CPU MuJoCo `201 preferences × 3 episodes` 评估。
5. 编译、稳态训练、Key 评估、checkpoint、CPU 评估和端到端时间分别记录到 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

## 测量与停止条件

- 分段 benchmark 使用 `jax.block_until_ready()`，不得为了计时向正式训练循环增加额外同步。
- 共享 GPU 必须记录设备、显存和其他进程占用；明显受竞争影响的数据只作诊断。
- 不采用单策略跨 GPU 同步训练；当前多 GPU 只做独立 seed 并行。
- 达到验收条件后停止扩展参数网格，进入正式复现实验。
