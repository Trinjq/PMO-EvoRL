# PMO-EvoRL 全局开发规则

入口索引：[`../AGENTS.md`](../AGENTS.md)。本目录指导：[`AGENTS.md`](AGENTS.md)。

## 1. 项目目标

本项目在 EvoRL/JAX 上实现并验证 GPU 原生的 PD-MORL。实现必须优先保持原 PD-MORL 源码的数据流和算法语义，再讨论并行规模与性能优化。

## 2. 事实来源优先级

1. 原 PD-MORL 官方源码的真实执行行为；
2. PD-MORL 论文对公式和设计意图的说明；
3. EvoRL 官方源码的接口与执行约束；
4. 项目内经测试确认的实现决策。

论文与原源码不一致时，以原源码行为为复现基线并明确记录差异；无法确认的内容标记为“待验证”，不得自行补全。

## 3. 架构边界

- `E:\PMO-EvoRL` 是代码、配置、测试和文档的主要编辑源。
- 实验室 `/home/qiuquanj/PMO-EvoRL` 用于 GPU 运行和保存大型实验产物。
- 源码通过 Git 同步；checkpoint、Replay Buffer、运行日志和大型结果不得提交仓库。
- 官方 EvoRL 作为依赖使用；PD-MORL 扩展优先放在本项目命名空间，不直接修改上游源码。
- 只有无法通过公共接口实现且有测试证明时，才允许维护上游补丁，并记录原因与对应 commit。

## 4. 算法语义

- 偏好 `w` 是 Actor/Critic/Q 网络的条件输入。
- 投影偏好 `wp` 不替代 `w`，只用于源码规定的 cosine/directional-angle 约束。
- 向量奖励、Vector Q 与标量化值 `w^T Q` 必须在命名和张量形状上明确区分。
- MO-TD3 选择 Twin Target Critic 时，先比较标量化值，再保留被选 Critic 的完整 Vector Q；不得逐元素拼接两个 Critic。
- HER 只重标记偏好，除非源码明确规定，否则不得重算状态、动作和向量奖励。
- 离散 MO-DDQN 与连续 MO-TD3 的 cosine/angle 机制分别实现，不共享未经验证的 Bellman 操作逻辑。

## 5. GPU 与 JAX 规则

- 训练主路径使用 JAX 数组与可 `jit` 的纯函数；避免在逐步采样路径中使用 NumPy、SciPy、Python 对象循环或主机回调。
- 并行环境、偏好、动作、向量奖励和 Replay 数据使用固定、显式的批量维度。
- 所有训练调度以原始环境 Transition 数为基准，不以会随 `num_envs` 改变含义的 Workflow iteration 为基准。
- 改变 `num_envs`、`rollout_length`、Batch 或更新次数时，记录 Critic 更新/原始 Transition 和 Actor 更新/原始 Transition。
- 共享 GPU 上默认关闭 JAX 显存预分配；使用 GPU 前遵守实验室分配规则。

## 6. 实现原则

- 先建立最小正确闭环，再增加 HER、Interpolator、多 GPU等复杂功能。
- 不为未出现的需求创建抽象层、兼容层或空模块。
- 公共接口使用清晰的数据结构和显式形状约定；禁止依赖隐藏的全局状态。
- 随机数必须显式传递 JAX PRNG key；不得混用未记录的 Python、NumPy随机状态。
- 所有配置项必须能追溯到源码基线、EvoRL需要或实验假设。

## 7. 验证顺序

每个训练功能至少按以下顺序验证：

1. 张量形状与 dtype；
2. 单步数值和终止逻辑；
3. `jit` 下的最小训练 step；
4. 无 NaN/Inf 的短程 GPU 冒烟测试；
5. 相同环境步数下与基线比较；
6. 最后比较环境步/秒及达到目标 Hypervolume 的墙钟时间。

非平凡逻辑至少保留一个可运行测试。性能提升不得以未经披露的环境、奖励或训练比例变化为代价。

## 8. 实验与证据

- 固定并记录 Git commit、依赖版本、GPU、随机种子、环境步数和关键超参数。
- 主要评价同时报告 Pareto Front、Hypervolume、相同环境步数表现和墙钟时间。
- 先完成单 GPU 正确性验证，再扩展多 GPU。
- 实验只为验证算法正确性、优势来源、目标场景价值或排除关键替代解释服务；避免无目的参数堆叠。

