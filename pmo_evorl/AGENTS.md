# `pmo_evorl` 模块指导

全局规则：[`../specs/GLOBAL_RULES.md`](../specs/GLOBAL_RULES.md)。

- 本包只放 PD-MORL 在 EvoRL 上的项目扩展，不复制或修改可通过 EvoRL/Brax 公共接口复用的功能。
- 所有训练主路径必须保持 JAX `jit`/`vmap` 可用，随机 key 显式传递。
- 新增高层子模块时，按根 [`AGENTS.md`](../AGENTS.md) 同步创建局部指导和索引。
