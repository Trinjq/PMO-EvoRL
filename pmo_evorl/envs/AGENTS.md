# `envs` 模块指导

上层指导：[`../AGENTS.md`](../AGENTS.md)。环境契约：[`../../specs/MO_WALKER2D_CONTRACT.md`](../../specs/MO_WALKER2D_CONTRACT.md)。

- 本目录只实现多目标环境语义；算法、Replay Buffer 和工作流不得放入此处。
- 奖励始终使用最后一维表示目标，不在环境内用偏好标量化。
- EvoRL 入口使用 `create_mo_walker2d_env`；它固定500步回合上限、保留终止时的原始观测，并把每个 episode 的偏好与状态一起作为结构化观测。
- 训练偏好必须复现原源码的1001点网格和10个连续子空间；`num_envs` 必须能被虚拟 Worker 数整除。
- 更改观测、动作、奖励、终止或时间尺度时，同步更新环境契约和最小回归检查。
