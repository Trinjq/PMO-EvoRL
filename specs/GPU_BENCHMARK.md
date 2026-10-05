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
