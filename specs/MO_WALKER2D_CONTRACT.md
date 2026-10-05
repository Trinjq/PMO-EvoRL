# MO-Walker2d 环境契约

全局规则：[`GLOBAL_RULES.md`](GLOBAL_RULES.md)。实现指导：[`../pmo_evorl/envs/AGENTS.md`](../pmo_evorl/envs/AGENTS.md)。

## 复现基线

以原 PD-MORL 仓库 `lib/utilities/morl/MOEnvs/moenvs/MujocoEnvs/walker2d.py` 的执行行为为基线：

- 观测：去掉水平位置的 `qpos` 与裁剪到 `[-10, 10]` 的 `qvel`，共17维。
- 动作：6维连续向量，执行前裁剪到 `[-1, 1]`。
- 速度奖励：`(x' - x) / dt + 1`。
- 能耗奖励：`4 - sum(a²) + 1`。
- 终止：高度严格位于 `(0.8, 2.0)` 且躯干角严格位于 `(-1, 1)`。
- 控制周期：`frame_skip=4`，XML 时间步为 `0.002`，因此 `dt=0.008`。
- 训练回合上限：500步。

## 当前实现边界

`MOWalker2d` 默认使用 MuJoCo MJX-JAX 在 GPU 上执行转换后的原 PD-MORL Walker2d 模型，并保留原始重置、动作裁剪、二维奖励、观测和终止公式。`mjx_impl=warp` 只用于通过本契约检查后的性能候选，严格基线默认仍为 `jax`。

原 XML 的 SHA-256 为 `09A3A898E7B5F7F053AD7BD8F917766488B9100FA0FC4CEBBA39044E8A46E6E0`。由于现代 MuJoCo 不再接受 `coordinate="global"`，本项目用 MuJoCo 2.3.3 加载原文件后通过 `mj_saveLastXML` 转换为局部坐标 MJCF。转换后文件的 SHA-256 为 `C1E13C86CEFB857F681802332539916F82387F1F628BC2CBC0C33B17027D4123`。

转换前后已比较 `qpos0`、质量、惯量、body/geom 位置、geom 尺寸与姿态、摩擦、阻尼、执行器 gear/ctrlrange、模型维度、时间步和积分器；数值差异不超过 `1.8e-15`。

MJX 与原 CPU MuJoCo 版本仍可能因求解器实现和浮点误差产生轨迹差异，因此当前声称是“模型参数与任务语义一致”，不声称“轨迹逐 bit 一致”。

完整 Pareto Front 使用原生 CPU MuJoCo 离线评估。该评估器在运行时强制检查同一转换后 XML、`nq=9`、`nv=9`、`nu=6`、RK4、`sim_dt=0.002`、`frame_skip=4` 和动作范围 `[-1,1]`，并复用本契约的17维观测、二维奖励、健康终止与500步上限。每个 episode 使用独立且写入结果文件的确定性种子。CPU与MJX固定状态单步回归要求终止一致、奖励绝对误差不超过 `1e-3`、观测绝对误差不超过 `1e-2`；这些阈值确认任务语义一致，不表示长轨迹逐值一致。

## 输入输出

- 裸环境 `reset(key) -> State`：`obs.shape == (17,)`，`reward.shape == (2,)`。
- `step(state, action) -> State`：`action.shape == (6,)`，输出二维 `reward=[speed, energy]`。
- EvoRL 训练入口输出 `obs.state [B,17]`、`obs.preference [B,2]` 和 `reward [B,2]`。
- 并行化后在最前方增加环境批次维；奖励最后一维始终是目标维。偏好在 episode 内保持不变，自动重置时从对应虚拟 Worker 的子空间重新采样。
- 非自动重置的 Key/GPU诊断评估使用 `VmapWrapper(vmap_step=True)`；正式 Pareto Front 仍由 CPU MuJoCo 离线评估。

## 最小验证

在实验室环境执行：

```bash
PYTHONPATH=. JAX_DEFAULT_MATMUL_PRECISION=highest python tests/test_mo_walker2d.py
```

检查重置/单步的 JIT 执行、形状、动作裁剪和二维奖励公式。
