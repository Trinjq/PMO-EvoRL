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

`MOWalker2d` 复用 Brax `Walker2d` 的 GPU 原生动力学、观测、随机重置和终止逻辑，覆盖动作裁剪与二维奖励。因此当前达到“交互公式一致”，不声称“物理轨迹逐步一致”。

原始 MuJoCo XML 使用 RK4；当前 Brax 后端及其 XML 与原实现不完全相同。是否可在 MJX 中无差异运行原 XML 标记为“待验证”；在验证前不将当前环境用于宣称与原论文的严格数值复现。

## 输入输出

- `reset(key) -> State`：`obs.shape == (17,)`，`reward.shape == (2,)`。
- `step(state, action) -> State`：`action.shape == (6,)`，输出二维 `reward=[speed, energy]`。
- 并行化后在最前方增加环境批次维；奖励最后一维始终是目标维。

## 最小验证

在实验室环境执行：

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest python tests/test_mo_walker2d.py
```

检查重置/单步的 JIT 执行、形状、动作裁剪和二维奖励公式。
