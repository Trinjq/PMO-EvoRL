# `tests` 目录指导

全局规则：[`../specs/GLOBAL_RULES.md`](../specs/GLOBAL_RULES.md)。

- 优先保留能直接用 Python 运行的最小检查，不为单个检查引入新测试框架。
- 非平凡逻辑至少覆盖形状、关键数值公式和 JIT 可执行性。
- 需要 GPU 的检查在实验室环境运行，不在本地仓库保存运行产物。
