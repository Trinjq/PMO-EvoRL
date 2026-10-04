# PMO-EvoRL 协作指导索引

本文件是仓库内所有指导文件和全局规范的入口。任何人或自动化代理在修改仓库前，都应先阅读本文件，再阅读目标路径向上最近的 `AGENTS.md` 及其引用的规范。

## 指导层级

1. 根目录 `AGENTS.md` 规定全仓库共同遵守的规则并维护索引。
2. `specs/` 保存稳定、全局适用的设计与开发规范。
3. 每个职责独立或抽象程度较高的功能模块目录必须有自己的 `AGENTS.md`。
4. 距离目标文件最近的 `AGENTS.md` 可以细化局部规则，但不得违背全局规范。
5. 同级规则冲突时，停止修改并先澄清；不得自行选择更方便的一条。

## 当前索引

| 范围 | 指导文件 | 内容 |
| --- | --- | --- |
| 全仓库 | [`AGENTS.md`](AGENTS.md) | 指导层级、索引和维护要求 |
| 全局规范目录 | [`specs/AGENTS.md`](specs/AGENTS.md) | `specs/` 内容边界与维护方式 |
| 全局开发规则 | [`specs/GLOBAL_RULES.md`](specs/GLOBAL_RULES.md) | 架构、实现、实验、验证与协作规则 |
| Walker2d 环境契约 | [`specs/MO_WALKER2D_CONTRACT.md`](specs/MO_WALKER2D_CONTRACT.md) | 原始 PD-MORL 环境语义与当前复现边界 |
| MO-TD3 网络契约 | [`specs/MO_TD3_NETWORK_CONTRACT.md`](specs/MO_TD3_NETWORK_CONTRACT.md) | 偏好条件 Actor 与 Twin Vector Critic 的输入输出 |
| 项目 Python 包 | [`pmo_evorl/AGENTS.md`](pmo_evorl/AGENTS.md) | PD-MORL 扩展的代码边界与验证规则 |
| 环境模块 | [`pmo_evorl/envs/AGENTS.md`](pmo_evorl/envs/AGENTS.md) | 多目标环境的契约和验证方法 |
| 测试 | [`tests/AGENTS.md`](tests/AGENTS.md) | 最小可运行回归检查 |
| 训练配置 | [`configs/AGENTS.md`](configs/AGENTS.md) | 超参数来源、采样/更新比例与运行配置 |

## 新模块落地规则

创建新的高层功能目录时，必须在同一次变更中：

1. 创建该目录的 `AGENTS.md`；
2. 写明模块职责、边界、关键数据契约、验证方法和相关规范；
3. 在上表加入索引；
4. 若与其他模块存在接口关系，在双方指导文件中互相引用；
5. 不为尚未实现的模块预建空目录或空指导文件。

模块内部仅用于归类的简单叶子目录，不强制单独创建指导文件；当其形成独立职责、数据契约或验证方法时再补充。

## 指导文件维护规则

- 指导文件只记录稳定约束、设计决策和可执行验证方式，不充当每日进度日志。
- 改变目录职责、公共接口或关键数据流时，必须同步更新对应 `AGENTS.md`。
- 新增全局约束时，写入 `specs/` 中合适的规范，并从本文件建立索引。
- 删除或移动规范、模块时，必须清理所有失效引用。
- 代码、配置、测试和文档以本仓库为唯一编辑源；实验室服务器用于运行与保存大型输出。
