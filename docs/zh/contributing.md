# 开源与贡献

ReMe 已开源，项目仓库托管于 GitHub：

**https://github.com/agentscope-ai/ReMe**

---

## 如何参与贡献

感谢你对 ReMe 的关注。ReMe 是一个面向 Agent 的、文件优先的自进化记忆系统，欢迎通过问题反馈、文档改进、测试补充、Bug
修复和新能力开发参与贡献。

如果是第一次本地运行，先看 [快速开始](./quick_start.md)。如果改动涉及运行时分层、Job、Step 或组件，先看
[ReMe 代码框架](./framework.md)；如果改动涉及 workspace 目录、frontmatter、wikilink 或 chunking，先看
[Memory as File](./memory_as_file.md)。

### 1. 开始之前

在投入实现前，建议先完成以下检查：

- 查看 [Open Issues](https://github.com/agentscope-ai/ReMe/issues)，确认是否已有相关问题或讨论。
- 如果相关 Issue 已存在且仍开放，请在评论中说明你想处理它，避免重复工作。
- 如果没有相关 Issue，请新建 Issue 描述背景、目标行为、可能的实现方向和影响范围。
- 对较大的功能变更，建议先和维护者对齐接口、配置、兼容性和测试策略，再提交实现。

### 2. 本地开发环境

ReMe 的核心代码位于：

- `reme/`：Python 包源码，包括配置、组件、服务、Job、Step、schema 和工具函数。
- `pyproject.toml`：项目元数据、依赖、可选依赖、命令入口和测试配置。
- `tests/`：单元测试和集成测试。

项目要求 Python 3.11 及以上。建议使用虚拟环境开发：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,full]"
pre-commit install
```

### 3. 代码开发范式

开发 ReMe 代码前，请先阅读 [ReMe 代码框架](./framework.md)。新增或修改核心能力时，应遵照其中描述的分层与调用链：

```text
CLI / Client -> Service -> Application -> Job -> Step -> Component / Workspace
```

也就是说：

- 面向用户或外部系统暴露的能力，优先通过 Job 编排，再由 Service 暴露为 CLI、HTTP 或 MCP 可调用接口。
- 可复用基础设施放在 `reme/components/`，通过 `BaseComponent.bind()` 声明组件依赖。
- 业务原子操作放在 `reme/steps/`，通过 `BaseStep.Ref` 访问 file store、agent wrapper、catalog、LLM 等组件。
- 请求、响应和持久化数据结构放在 `reme/schema/` 或 `reme/enumeration/`，不要把隐式结构散落在 Step 内部。
- 配置驱动的默认行为写入 `reme/config/default.yaml`，并保持默认配置可启动、可测试。

新增 Step 或 Job 时，特别注意以下约定：

- 使用 `@R.register("<backend_name>")` 注册实现，注册名应稳定、清晰，并与配置中的 `backend` 对齐。
- 新增 Step 文件后，确认所在包的 `__init__.py` 会 import 该模块，否则注册表不会加载它。
- Step 只处理单个业务原子操作；跨步骤流程应放在 Job 配置或专门的编排 Step 中。
- Job 负责组合 Step，并决定普通、流式、后台或定时执行方式；是否对外暴露由 `enable_serve` 控制。
- Step 需要组件时优先使用 `BaseStep.Ref`，不要在 Step 内重新构造全局组件或绕过 `ApplicationContext`。
- 涉及文件、索引、图谱、front matter、wikilink 的行为，应保持 workspace-relative 路径语义一致。
- 新能力应补充 `tests/unit/` 中的快速测试；跨组件、LLM、embedding 或服务行为再放入 `tests/integration/`。

### 4. 代码与文档修改建议

根据改动类型选择合适的入口：

| 改动类型       | 主要位置                                                | 建议                                                                        |
|------------|-----------------------------------------------------|---------------------------------------------------------------------------|
| 配置或启动行为    | `reme/config/`、`reme/application.py`、`reme/reme.py` | 保持默认配置可运行，避免破坏现有 CLI、HTTP 和 MCP 入口                                        |
| 组件能力       | `reme/components/`                                  | 优先复用 `BaseComponent`、registry 和上下文对象                                      |
| Job 或 Step | `reme/components/job/`、`reme/steps/`                | 遵照 [ReMe 代码框架](./framework.md) 的 Job -> Step 范式，保持请求、响应 schema 清晰，并补充对应测试 |
| 数据结构       | `reme/schema/`、`reme/enumeration/`                  | 注意序列化兼容性和已有 front matter、wikilink 语义                                      |
| 工具函数       | `reme/utils/`                                       | 保持函数边界小，并用单元测试覆盖边界情况                                                      |
| 用户文档       | `docs/zh/`、`README.md`                              | 当用户可见行为变化时同步更新文档                                                          |

如果改动涉及 LLM、embedding、外部服务、文件监听或后台任务，请同时说明依赖条件、失败行为和本地验证方式。

### 5. 提交信息格式

建议遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，以保持历史记录清晰。

格式：

```text
<type>(<scope>): <subject>
```

常用类型：

- `feat`：新功能
- `fix`：Bug 修复
- `docs`：仅文档
- `style`：代码风格调整，不改变行为
- `refactor`：重构，不修复 Bug 也不添加功能
- `perf`：性能改进
- `test`：添加或更新测试
- `chore`：构建、工具或维护工作

示例：

```bash
feat(search): add link expansion option
fix(file-graph): handle pending wikilinks after move
docs(memory): update auto memory guide
test(config): cover default yaml parsing
chore(pre-commit): update lint hooks
```

### 6. Pull Request 标题

PR 标题建议使用相同格式：

```text
<type>(<scope>): <description>
```

要求：

- 类型使用 `feat`、`fix`、`docs`、`test`、`refactor`、`chore`、`perf`、`style`、`build` 或 `revert`。
- 作用域使用小写字母、数字、连字符或下划线。
- 描述保持简短，说明这次 PR 的实际效果。

示例：

```text
feat(auto-memory): persist source conversation metadata
fix(markdown): keep wikilink aliases during edit
docs(zh): add contribution guide
```

### 7. 提交前检查

提交或发起 PR 前，请至少运行：

```bash
pre-commit run --all-files
pytest
```

如果只改了局部代码，可以先运行更小范围的测试：

```bash
pytest tests/unit/test_search_step.py
pytest tests/unit/test_reme_cli.py
```

如果 `pre-commit` 自动修改了文件，请提交这些修改后重新运行检查，直到全部通过。

当前 pre-commit 配置包括 YAML/TOML/JSON 检查、私钥检测、尾随空格检查、`black`、`flake8`、`pylint` 和 `pyroma`。代码格式主要遵循：

- `black --line-length=120`
- `flake8 --max-line-length=120`
- `pylint --max-line-length=120`

部分集成测试可能依赖 LLM、embedding 或外部服务配置。若无法在本地完整运行，请在 PR 描述中说明跳过原因和已完成的替代验证。

### 8. 测试要求

请根据改动风险补充测试：

- 修复 Bug 时，优先添加能复现问题的回归测试。
- 新增 Step、Job 或组件时，至少补充核心路径和失败路径测试。
- 修改索引、图谱、wikilink、front matter、文件读写等共享逻辑时，补充边界用例。
- 修改 CLI、服务或配置解析时，覆盖用户可见入口。
- 文档-only 改动通常不需要新增测试，但仍建议运行 `pre-commit run --all-files`。

测试文件按现有结构放置：

- `tests/unit/`：无需真实外部服务的快速测试。
- `tests/integration/`：跨组件或依赖外部配置的集成测试。

### 9. 文档贡献

当你的修改会影响用户如何安装、配置、调用或理解 ReMe 时，请同步更新文档。

文档位于：

```text
docs/
```

建议文档保持：

- 标题明确，直接说明能力或流程。
- 命令可以复制运行。
- 涉及路径时使用仓库内真实路径，例如 `reme/config/default.yaml`、`reme/steps/`、`tests/unit/`。
- 涉及默认行为时，以当前代码和 `pyproject.toml`、默认配置为准。

---

## 获取帮助

- Bugs 和功能请求：[GitHub Issues](https://github.com/agentscope-ai/ReMe/issues)
- 项目主页：[GitHub Repository](https://github.com/agentscope-ai/ReMe)
- 文档站点：[https://reme.agentscope.io/](https://reme.agentscope.io/)

---

感谢你为 ReMe 做出贡献。你的改进会帮助 Agent 的长期记忆更可读、可控、可维护。
