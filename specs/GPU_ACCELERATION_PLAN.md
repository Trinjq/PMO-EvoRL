# PD-MORL GPU 加速执行计划

入口索引：[`../AGENTS.md`](../AGENTS.md)。全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。实测结果：[`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)。

## 目标与验收

唯一优化目标是在保持证据质量时缩短从进程启动到 CPU Pareto 指标写出的端到端墙钟时间。最终速度优先版本必须满足：

- 三个随机种子的平均 Hypervolume 不低于严格基线的95%；
- 任一随机种子的 Hypervolume 不低于对应基线的90%；
- 端到端墙钟时间至少提升1.5倍；
- 环境、奖励、终止条件、偏好输入、HER语义和最终 CPU MuJoCo 评估协议不变。

正在运行的实验必须自然完成，不得因加速开发中断或修改。`configs/mo_td3_walker2d.yaml` 始终保留严格源码比例；算法级加速只能使用独立配置或命令行覆盖，并明确标记为变体。

## 测量门

1. 所有性能数字使用 [`../benchmark.py`](../benchmark.py)，正式 capacity 为 `2,000,000`。
2. 每个模式先 warm-up，再连续测三次并报告中位数；设备、依赖版本、commit 和有效 buffer 占用必须随结果保存。
3. JAX 分段计时必须 `block_until_ready()`；正式训练循环不为性能归因增加同步，只记录端到端与 checkpoint 时间。
4. 基础测量包括 `full`、`env-only`、`learner-only` 和 `key-eval`。`learner-only` 是训练路径减去 env-only 的估计，并结合一次 JAX trace 判断 kernel 占比。
5. GPU 训练和 benchmark 必须强制 CUDA；共享 GPU 存在明显竞争时数据作废。

## 语义等价筛选

1. 非自动重置评估开启 batched `vmap`；RBF 系数仅在初始化和 Key 更新时求解。
2. 先分别测 vmap 和 RBF cache，再测组合版本；数值回归失败或没有稳定收益的复杂改动不保留。
3. 组合版本比较320和640环境。只有640的完整训练吞吐比320至少高15%，才继续测1280。
4. 对入选规模采集一次 JAX trace，确认环境、learner、Key评估和主机同步占比。
5. Warp 已因640环境 contact overflow 且吞吐低于JAX而淘汰；不增加 `mjx_impl` 正式配置，也不搜索 graph mode/contact buffer 等次级参数。

最快且通过数值检查的语义等价配置成为新 B0。

## 算法候选与晋级

- Key评估占端到端时间不超过10%时，保留 `3 episodes / interval 1`，不搜索 Key 参数。
- Key评估超过10%时，依次测试 `1 episode`、`interval 5`；优化后仍超过20%才测试在线 CPU Key评估。
- learner占比超过50%时，测试 `critic/transition=0.5`、`actor/transition=0.05`，并明确标记为算法加速变体。
- 每个候选先运行100万步、seed 1；出现 NaN、训练失败或 Hypervolume 低于新 B0 的90%立即淘汰。
- 最快两个候选运行300万步、seeds 1/2；平均 Hypervolume 至少为基线95%，且单 seed 不低于对应基线90%。
- 最终候选运行1000万步、seeds 1/2/3，每张可用 GPU 运行一个独立 seed；所有 checkpoint 统一执行 CPU `201 preferences × 3 episodes` 评估。

达到质量和1.5倍端到端加速门槛后停止扩展参数网格，进入正式复现与论文结果整理。

## 固定边界

- 完整 Pareto Front 只使用 CPU MuJoCo 离线协议；GPU Pareto 仅作回归诊断。
- 不采用单策略跨 GPU 同步训练；多 GPU 只并行独立 seeds。
- 编译、稳态训练、Key评估、checkpoint、CPU评估和端到端时间记录到 [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md)，不新建重复的状态或证据文件。
