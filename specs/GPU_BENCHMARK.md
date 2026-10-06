# 单 GPU 吞吐基准（初测）

全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。运行工具：[`../benchmark.py`](../benchmark.py)。

## 目的与口径

初测回答：在不降低学习强度时，增加 GPU 并行环境数能否提高训练 step 吞吐。所有配置固定：

- 每条原始 transition 对应1次 Critic更新；
- 每条原始 transition 对应0.1次 Actor/Target更新；
- Batch size为256；
- 每个配置先完成一次编译和 warm-up，再计时10个训练 iteration；
- 计时包括 MJX 采样、Replay sampling、Critic/Actor反向传播和 Target更新，不包括首次 XLA 编译。

初测使用 `100,000` replay capacity、一次 warm-up 和一次正式计时，不包含 Key 评估，因此下表不是端到端训练结论。

## 2026-10-04 初测结果

硬件为实验室 NVIDIA GeForce RTX 4090（GPU 2），使用提交 `1e60ae9`。服务器为共享状态，因此这些数字用于选择候选规模，不作为论文最终性能证据。

| 并行环境数 | `num_updates_per_iter` | 原始 transition/s | 相对10环境 | Critic/transition | Actor/transition |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 159.15 | 1.00× | 1.0 | 0.1 |
| 80 | 8 | 665.30 | 4.18× | 1.0 | 0.1 |
| 160 | 16 | 853.46 | 5.36× | 1.0 | 0.1 |
| 320 | 32 | 1410.86 | 8.86× | 1.0 | 0.1 |
| 640 | 64 | 1704.19 | 10.71× | 1.0 | 0.1 |
| 1280 | 128 | 1995.60 | 12.54× | 1.0 | 0.1 |

## 当前选择

首轮等环境步数训练采用320环境。它保留更新比例，吞吐约为基线的8.86倍，且学习预填充因批量粒度只从1540条变为1600条。640环境会在1920条后开始学习，1280环境会在2560条后开始学习，早期训练调度偏差更大。

训练入口对大于10的并行环境数把 `fold_iters` 上限设为100，使320环境每个外层 fold 最多跨越32,000条 transition。最终停止条件依据实际累计的原始 transition，而不是 EvoRL 上游混用的 fold计数；最后一折会完成评估并强制保存 checkpoint。

动态 Interpolator 的episode计数在每个 fold 返回主机后检查；500步 Key评估单独 JIT执行，不嵌入逐step训练扫描。这样避免 XLA 把低频评估分支并入100步训练图。检查粒度最多为100个iteration；若实验需要逐episode严格触发，可把 `fold_iters` 调小。

折叠训练把输入 state donation 给 JAX；调用方只使用返回的新 state，使 XLA 可以原位复用大容量 Replay Buffer 的设备内存，避免每个step复制整棵200万容量 buffer PyTree。

这些结果只证明稳态吞吐提高，不证明达到相同 Hypervolume 所需的环境步数不变。正式结论必须比较相同随机种子、相同原始环境步数下的 Pareto Front、Hypervolume和墙钟时间。

## Pareto 评估修正

初版评估即使所有 episode 已终止，仍固定执行500步物理模拟。2026-10-04 在同一台共享 RTX 4090 上，以201个偏好、每个偏好1个episode和批量201复测：改为批次内全部episode终止后立即退出，首次评估从1641.71秒降至229.18秒，稳定评估从1609.28秒降至199.05秒，Hypervolume与Sparsity保持一致。但32万步训练后的策略因长回合使在线GPU终评超过37分钟，因此完整 Pareto 评估最终从训练循环移除，改为保存 checkpoint 后使用原生 CPU MuJoCo 离线执行。上述结果只用于工程选型，不作为算法性能证据。

同一个32万步 checkpoint 在当前原生 CPU MuJoCo 离线协议下，201偏好×1 episode耗时7.06秒，201偏好×3 episode耗时23.06秒。评估器强制校验与MJX相同的模型和任务契约，将GPU checkpoint显式重映射到CPU，并在结果中保存每个episode的独立确定性种子。CPU与MJX使用不同数值后端和随机流，轨迹及指标不作逐值等同解释；所有正式 checkpoint 统一使用该CPU协议评估。

## 三 GPU 执行边界

`run_multi_gpu.py` 为每张 GPU 分配一个独立随机种子。每个进程仍在指定 GPU 上执行批量 MJX 环境、Replay 采样和网络更新。这能在不改变单次 PD-MORL 训练的 Batch 与梯度语义的前提下，提高整组实验的总吞吐。

该方式属于种子级并行，不是一个策略的同步数据并行。当前未启用同步三 GPU 训练：已安装的 EvoRL 多设备路径与 JAX 0.10.2 不兼容；同时，改变每卡 Batch 会改变有效优化 Batch。只有在保持算法语义的单策略缩放基准确认存在墙钟收益后，才加入同步数据并行。

## 2026-10-05 新测量协议

[`../benchmark.py`](../benchmark.py) 默认使用正式 `2,000,000` capacity、一次预热和三次同步计时并报告中位数，同时记录有效 buffer 占用、设备、依赖版本和 Git commit。

[`../benchmark_components.py`](../benchmark_components.py) 用于 Step 1 的分模块 microbenchmark；它在较小 capacity 下分别计时 rollout、原始 Replay add、eager/lazy HER add、sampling、Critic forward/update、Actor update、RBF projection、Key evaluation 和完整训练 iteration。该脚本只生成测量 JSON，不替代正式 `benchmark.py` 的 2M capacity 端到端门。

- `full`：一个训练 fold 加一次 Key 更新，用于测量 Key 边界上的完整路径；
- `train`：不含 Key 更新的训练路径；
- `env-only`：固定零动作的批量 MJX step；
- `learner-only`：`train - env-only` 的估计值，明确包含策略推理和 Replay 开销；EvoRL 当前没有可复用的独立 learner 入口，不复制其 TD3 更新循环；
- `key-eval`：三关键偏好评估和 RBF 更新；
- `pareto-eval`：仅作 GPU 回归诊断，正式指标仍来自 CPU MuJoCo。

每个模式先编译和 warm-up，再对三个连续样本取中位数。可用 `--profile-dir` 为首个正式训练样本生成 JAX trace。新协议的实测数字只在实验室 GPU 空闲且结果完成后追加到本文件。

## 2026-10-05 语义等价优化诊断

使用提交 `b1bbe28`、JAX 0.10.2、MuJoCo/MJX 3.14.0、Optax 0.2.8，在共享 RTX 4090 GPU 2 上关闭显存预分配测试。GPU 上同时存在其他用户进程，因此本节只用于执行筛选门，不作为论文最终性能证据。

非自动重置评估由 `lax.map` 改为 `vmap` 后，三关键偏好、每个偏好3个episode的单步观测最大绝对差为 `3.91e-8`，奖励和终止完全一致。Key evaluation 三次分别为2.016、2.031和2.485秒，中位数2.031秒；旧 B0 七次 Key evaluation 的中位数为170.711秒，因此诊断加速约84倍。按旧 B0 的7次更新频率估算，新 Key evaluation 在100万步训练中约占14秒，不再达到10%的参数搜索门槛。

所有 train 测试使用正式2,000,000 replay capacity、10个step、一次预热和三次同步计时，并保持每transition 1.0次Critic和0.1次Actor更新：

| 环境数 | 初始 Buffer | 中位时间/s | transition/s | 相对前一档 |
| ---: | ---: | ---: | ---: | ---: |
| 320 | 1600 | 2.084 | 1535.8 | — |
| 640 | 1920 | 3.207 | 1995.6 | +29.9% |
| 1280 | 2560 | 5.673 | 2256.4 | +13.1% |

640相对320超过15%晋级门；1280相对640只有13.1%，停止继续扩大环境数。当前1M质量筛选候选采用640环境、3次Key episode、间隔1和原始更新比例。

320环境的 env-only 中位数为1.393秒，train中位数为2.084秒，差值 learner-path 估计为0.690秒、约占33%。首个env样本30.823秒为共享GPU/额外编译异常值，由三样本中位数自然排除。learner未超过50%门槛，因此当前不测试减半更新强度。

强制每10个训练step执行一次Key更新的 `full` 最坏边界中位数为5.394秒、593.3 transitions/s；这不是实际训练频率，只用于确认Key边界同步后的上限。

### Warp 候选淘汰

MJX-Warp 1.17.0 的单环境 build/forward/4-substep JIT 与小批量 Walker2d 契约检查可以运行；但640环境训练持续报告默认 contact buffer 的 broadphase/narrowphase overflow，已不满足环境正确性门槛。该无效运行的中位吞吐也只有1782.1 transitions/s，比同卡 JAX 的1995.6低10.7%。因此停止调整 contact buffer、graph mode等Warp专用参数，撤回正式配置接口，当前候选继续使用MJX-JAX。

## 2026-10-05 语义等价 B0：100万步质量门

在同一台共享服务器上，以 seed 1 运行640环境候选至1,025,920条原始 transition。训练过程中 Actor/Critic loss 均为有限值。CPU MuJoCo `201 preferences × 3 episodes` 评估得到 Hypervolume 3,667,026.65、Sparsity 2,667.80；旧320环境 B0 的对应结果为 Hypervolume 3,478,076.76、Sparsity 1,243.05。因此新候选 Hypervolume 为旧 B0 的105.4%，通过90%淘汰门。

新候选 build、初始化、训练和CPU评估分别耗时71.30、118.48、689.54和42.83秒，完整可观测流程共922.15秒；旧 B0 对应为193.69、488.68、2811.29和103.64秒，共3597.30秒。共享 GPU 条件下诊断加速为3.90倍，超过最终1.5倍目标，但正式结论仍需以最终三 seed 实验为准。候选已晋级300万步 seeds 1/2。

## 2026-10-05 语义等价 B0：300万步晋级门

640环境候选的 seeds 1/2 均完成3,009,920条原始 transition，训练耗时分别为2291.47和2092.14秒，Actor/Critic loss 均保持有限。统一 CPU MuJoCo `201 preferences × 3 episodes` 评估得到：seed 1 的 Hypervolume 为4,172,738.08、Sparsity为1,168.51；seed 2 的 Hypervolume 为4,922,693.94、Sparsity为1,867.46。平均 Hypervolume 为4,547,716.01，是1M新 B0 的124.0%，两个 seed 均高于1M质量锚点。

两项CPU评估并行运行并争用主机CPU，各耗时约168秒，因此该评估耗时只记录为本次实际墙钟，不用于单进程性能归因。候选质量稳定，已晋级最终1000万步 seeds 1/2/3。

## 2026-10-06 优化方案实验室验证（2M raw transitions）

本节记录优化方案当前提交 `d9fed35` 的实验室验证结果。环境为 JAX 0.10.2、MuJoCo/MJX 3.14.0、Optax 0.2.8；分模块 profiling 使用 GPU 0，正式训练使用 GPU 1，均设置 `CUDA_VISIBLE_DEVICES`，实验期间服务器仍为共享状态。所有质量结果均来自实验室 CPU MuJoCo 的 `201 preferences × 3 episodes` 离线协议；训练实际完成 `3,200 iterations / 2,049,920 raw transitions`。

### Step 1 分模块 profiling

`benchmark_components.py` 使用 capacity 100,000、10 steps、3 个稳态样本，HER 从第 0 个 transition 生效。`full_iteration` 和 `rollout` 的稳态中位数如下；完整 JSON 保存在实验室工作树 `outputs/optimization_20261006/logs/components_*_her0_capacity.json`。

| 并行环境数 | rollout/s | full iteration/s | key evaluation/s |
| ---: | ---: | ---: | ---: |
| 10 | 0.668 | 0.060 | 3.018 |
| 80 | 1.383 | 0.133 | 2.605 |
| 160 | 1.305 | 0.150 | 2.666 |
| 320 | 2.408 | 0.305 | 3.166 |
| 640 | 3.823 | 0.436 | 3.309 |
| 1280 | 7.677 | 0.693 | 2.601 |

Replay add/sample 均保持在毫秒级；随着并行环境数增加，主要增长来自 MJX rollout 与完整 learner iteration，而不是 Replay Buffer 操作。capacity 100,000 下，eager HER 分配约 18.0 MB，Lazy HER 的等 raw-equivalent capacity 为 25,000、分配约 4.525 MB，内存减少约 74.9%。

### 2M 质量与墙钟门

以下三组训练使用 640 环境、seed 1、相同 raw transition 预算和相同 CPU 评估协议。baseline 为 eager HER、逐次 sample；source-parity 为 Lazy HER + cached RBF + `sample_many()`；GPU-oriented 进一步使用 batch 1024、critic UTD 0.25。

| 版本 | 训练时间/s | HV | Sparsity | 结论 |
| --- | ---: | ---: | ---: | --- |
| baseline | 1938.60 | 4,504,328.40 | 880.73 | 参考 |
| source-parity optimized | 1854.65 | 4,528,864.53 | 2,245.78 | 不通过：Sparsity 明显恶化 |
| GPU-oriented | 1789.72 | 3,434,573.92 | 2,690.98 | 不通过：HV 与 Sparsity 均恶化 |

因此当前实现可以保留 Lazy HER 的内存/工程实验代码和 profiling 工具，但不能把 source-parity 或 GPU-oriented 配置宣称为已验证的默认训练方案。后续若继续推进，应先用固定随机流或逐项消融隔离 Lazy HER 与 `sample_many()` 对 Pareto 解分布的影响，再重新申请质量门；在此之前，正式默认配置保持 baseline 语义。
### Fixed-seed Lazy HER 消融

在 `c011d84` 上用实验室 GPU 0 完成同一 2M raw-transition 训练，训练耗时 1299.47 s；CPU MuJoCo 评估得到 HV 4,273,817.98、Sparsity 1,939.32。相较 baseline，HV 下降且 Sparsity 恶化，因此该实现通过了 replay/workflow 回归，但未通过 Step 2 的学习质量门；正式默认配置继续保持 baseline 语义。

同一提交的 640-env stage profile 显示 lazy buffer capacity 为 25,000、占用约 4.625 MB（eager 为 18.0 MB）；rollout、full iteration、key evaluation 的稳态中位数分别为 2.264 s、0.477 s、2.055 s。该内存收益不能抵消学习质量门失败。

### 2026-10-06 v2 Step 1：640-env eager-HER 基线

提交 `04aa6fe` 在实验室 GPU 上完成了当前 source-compatible eager-HER 基线：640 个环境、batch 256、Critic/transition 1.0、Actor/transition 0.1、seed 1、3,200 iterations，共 `2,049,920` 条 raw transitions。实验环境为 JAX 0.10.2、MuJoCo/MJX 3.14.0、Optax 0.2.8；训练墙钟为 `2,375.86 s`。

训练配置中的 Pareto 网格仍为 `pareto_step_size=0.005`，即201个 preference；为满足 v2 评估协议，最终 checkpoint 另在实验室 CPU MuJoCo 上以 `step-size=0.001`、1001个 preference、每个 preference 3个 episode 评估。评估墙钟为 `533.18 s`，其中实际模拟与指标计算为 `531.06 s`。结果如下：

| 指标 | 结果 |
| --- | ---: |
| Hypervolume（3次评估均值） | 4,508,179.11 |
| Sparsity（3次评估均值） | 387.44 |
| ParetoCount（3次评估均值取整） | 238 |
| source HV（1001个 mean-return 点） | 4,485,636.28 |
| source Sparsity | 303.66 |
| source ParetoCount | 277 |
| ParetoCount per repeat | 248 / 226 / 241 |

该结果冻结了 v2 后续逐项实验的质量与墙钟参考。评估输出保存在实验室工作树 `outputs/pmo_phsl_v2/train/step1_baseline_640_2m/eval_cpu_1001/`；训练和评估日志分别为 `outputs/pmo_phsl_v2/logs/step1_baseline_640_2m.log` 与 `step1_baseline_640_2m_eval_1001.log`。其中 `source_*` 指标用于记录原始 1001-point preference 网格上的 mean return 前沿，不能与每重复评估后再取 Pareto 前沿的指标混用。

### 2026-10-06 v2 Step 2：logical-group random warm-up

提交 `41a81c9` 在相同 640-env、seed 1、2M raw-transition协议下恢复了10个 logical group 的随机 warm-up。每组在达到10,000条 group transitions 前使用均匀随机动作，之后使用 `actor(s,w)+exploration noise`；最终日志记录每组 `10,048` 条随机 transition，总随机 transition 为 `100,480`，policy transition 为 `1,947,520`，random action fraction 为 `4.90%`。这与 `10 × 10,000` 的 source 语义一致，超出部分来自每次 rollout 的64条 group transitions粒度。

训练完成 `3,200 iterations / 2,049,920 raw transitions`，墙钟为 `2,614.16 s`。同一 checkpoint 在实验室 CPU MuJoCo 上以1001个 preference、每个preference 3个episode评估，评估墙钟为 `428.48 s`，结果如下：

| 指标 | Step 1 baseline | Step 2 warm-up | 变化 |
| --- | ---: | ---: | ---: |
| Hypervolume | 4,508,179.11 | 4,380,114.58 | -2.84% |
| Sparsity | 387.44 | 319.30 | -17.6% |
| ParetoCount | 238 | 222 | -6.7% |
| source HV | 4,485,636.28 | 4,348,070.88 | -3.07% |
| source Sparsity | 303.66 | 228.98 | -24.6% |
| source ParetoCount | 277 | 298 | +7.6% |

Step 2 改善了 mean-return 网格的 source Sparsity，并提高了 source ParetoCount，但重复评估的 HV 和 ParetoCount 低于 Step 1，因此不能宣称整体质量优于 baseline。该结果作为 Step 3 logical-group interpolator 实验的新的可比锚点；训练输出保存在 `outputs/pmo_phsl_v2/train/step2_warmup_640_2m/`，日志为 `outputs/pmo_phsl_v2/logs/step2_warmup_640_2m.log` 和 `step2_warmup_640_2m_eval_1001.log`。

### 2026-10-06 v2 Step 3：logical-group interpolator control

提交 `c0dc2bc` 在 Step 2 warm-up 基础上，将 interpolator trigger 改为10个 logical group的聚合 episode count：每组对其 physical lanes 的累计 episode数求和后除以 lanes数取 floor，并以10组中的最小值触发 `key_update_interval=1`。训练日志显示 group count 会自然分化，例如最终为 `[9,9,9,10,10,10,10,11,11,12]`；这证明 trigger 不再使用 `episode_count.min()` 的640-lane物理语义。最终计数为 `key_evaluation_count=8`、`key_replacement_count=0`、`interpolator_refit_count=8`，Actor/Critic loss 全程 finite。

训练完成 `3,200 iterations / 2,049,920 raw transitions`，墙钟为 `1,844.91 s`。1001个preference、每个preference 3个episode的实验室 CPU MuJoCo 评估耗时 `325.43 s`，结果如下：

| 指标 | Step 1 baseline | Step 3 logical-group control | 变化 |
| --- | ---: | ---: | ---: |
| Hypervolume | 4,508,179.11 | 4,616,349.96 | +2.40% |
| Sparsity | 387.44 | 423.97 | +9.4% |
| ParetoCount | 238 | 154 | -35.3% |
| source HV | 4,485,636.28 | 4,519,039.92 | +0.75% |
| source Sparsity | 303.66 | 209.19 | -31.1% |
| source ParetoCount | 277 | 191 | -31.0% |

Step 3 提高了重复评估 HV 和 source Sparsity，但重复 ParetoCount 明显下降，不能据此宣称全面优于 baseline；其主要证据是修复了 logical-group episode 语义并控制了 key 更新频率。训练输出保存在 `outputs/pmo_phsl_v2/train/step3_logical_group_interpolator_640_2m/`，日志为 `outputs/pmo_phsl_v2/logs/step3_logical_group_interpolator_640_2m.log` 和 `step3_logical_group_interpolator_640_2m_eval_1001.log`。

### 2026-10-06 v2 Step 5：eager-HER source-compatible insertion order

提交 `403e9e6` 将 eager HER 的写入顺序改为每条 raw transition 的 `原始、HER1、HER2、HER3`，并用实验室网络回归检查了原始 transition 与其 HER entries 的分组顺序及 preference 归一性。Step 4 的 global-L2 clipping仍由同一 optimizer chain提供：`clip_by_global_norm(100.0) → Adam`；裁剪算子的 raw/clipped norm 回归也在提交 `5e2cb6e` 中通过。

在保留 Step 2 warm-up 和 Step 3 logical-group interpolator control 的条件下，完成 `3,200 iterations / 2,049,920 raw transitions`，训练墙钟为 `1,603.06 s`。1001个preference、每个preference 3个episode的 CPU MuJoCo 评估耗时 `341.19 s`：

| 指标 | Step 3 control | Step 5 interleaved HER | 变化 |
| --- | ---: | ---: | ---: |
| Hypervolume | 4,616,349.96 | 4,867,783.38 | +5.45% |
| Sparsity | 423.97 | 3,285.97 | +675% |
| ParetoCount | 154 | 159 | +3.2% |
| source HV | 4,519,039.92 | 4,834,480.83 | +6.98% |
| source Sparsity | 209.19 | 2,847.02 | +1261% |
| source ParetoCount | 191 | 196 | +2.6% |

交错插入提高了 HV 和 ParetoCount，但使 Sparsity 大幅恶化，因此通过了 insertion-order 语义回归，未通过 Pareto 覆盖质量门。该版本保留为 source-faithful 消融结果；正式结论不得把它描述为无条件优于 Step 3。训练输出保存在 `outputs/pmo_phsl_v2/train/step5_eager_her_interleaved_640_2m/`，日志为 `outputs/pmo_phsl_v2/logs/step5_eager_her_interleaved_640_2m.log` 和 `step5_eager_her_interleaved_640_2m_eval_1001.log`。

质量门失败后，正式 fast 配置将 `interleave_her` 固定为 `false`，恢复 Step 1/Step 3 的 baseline 写入顺序；交错版本仍可通过 `train.py --interleave-her` 显式复现实验，不再无条件进入默认训练。

### v2 Step 4 与 Step 6：已有实现的直接验证

Step 4 的 global-L2 clipping 在当前 workflow 中由 `clip_by_global_norm(100.0)` 明确包在 Actor/Critic 共用 optimizer chain 的最前端。提交 `ac8005d` 增加了 raw/clipped gradient norm 统计；实验室 64k raw-transition 短训练日志记录到 Critic 的累计 raw norm `1297.10`，而 clipping 阈值为100，说明统计路径确实覆盖了需要裁剪的梯度；该短训练中 loss finite。Step 4 的 2M质量结果已由 Step 5 的同一 optimizer chain 共同覆盖，未另行重复一份仅改日志的2M训练。

Step 6 的数学实现早于本轮计划：`fit_linear_rbf` 只在初始化或 interpolator 更新时求解，Actor/Critic loss 只调用缓存系数的 `evaluate_linear_rbf`。提交 `930f017` 将实验室网络回归扩展为100个随机 simplex preference，并同时比较 projection 与 query gradient，断言最大绝对误差不超过 `1e-5`；同一回归还检查 Actor/Critic loss 为 finite。由于 cached-RBF 已经包含在 Step 1 的冻结 baseline 中，Step 1 的2M结果就是该实现的质量锚点；没有再重复一个数学上相同、无代码差异的2M训练。

### v2 Step 7：evaluation 双轨指标与可追溯输出

提交 `04aa6fe` 已将 CPU evaluator 升级为每个 repeat 独立计算 HV、Sparsity 和 ParetoCount，同时对 `mean return per preference` 计算 `source_hv`、`source_sparsity` 和 `source_pareto_count`。1001-point正式评估已在 Step 1、Step 2、Step 3 和 Step 5 完成；每次结果均保存三次 repeat 指标及其 ParetoCount。

提交 `d51ec45` 进一步在 `returns.npz` 中保存 `preferences`、`evaluation_preference_grid`、`returns_per_repeat`、`mean_returns`、`pareto_mask`、`pareto_returns`、`objective_min/max/mean`、episode lengths 和 episode seeds；`metrics.json` 同步保存 objective statistics。实验室 artifact check 已验证这些数组实际存在（例如 `pareto_returns` 形状为 `(39,2)`，`objective_min` 形状为 `(2,)`），因此 Step 7 的 collapse 诊断和 preference→return 追溯要求均已覆盖。
