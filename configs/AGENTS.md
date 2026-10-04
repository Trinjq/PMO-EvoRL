# `configs` 目录指导

全局规则：[`../specs/GLOBAL_RULES.md`](../specs/GLOBAL_RULES.md)。算法契约：[`../specs/MO_TD3_NETWORK_CONTRACT.md`](../specs/MO_TD3_NETWORK_CONTRACT.md)。

- 配置必须注明参数来自 PD-MORL 源码、EvoRL 运行要求或实验假设。
- 调整并行环境数时，必须保持并记录每条原始 transition 对应的 Critic 与 Actor 更新次数。
- `learning_start_timesteps` 与 `her_start_timesteps` 含义不同，禁止合并。
- 正确性基线配置先保持源码训练比例；加速配置必须另建并通过等环境步数实验验证。
