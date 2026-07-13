# ReMe 文档设计

本文定义 ReMe 文档的内容边界和维护方式。目标是让文档保持精简、稳定，并适合用户与 AI coding agent 快速理解。

## 两类文档，两种职责

| 位置 | 用途 | 是否部署 |
|---|---|---|
| `ReMe/docs/` | README 引用的中英文补充说明和图片 | 否 |
| `agentscope-ai/docs/reme/<version>/` | 面向用户的中英文产品文档 | 是 |

ReMe 仓库维护 `docs/en/`、`docs/zh/` 中供 README 直接引用的页面，但不把它们作为网页部署来源。网站内容、发布、版本选择、
导航和重定向都由统一文档仓库负责。

## 内容原则

### Concepts 只讲理念

Concepts 应解释 ReMe 为什么这样设计，而不是逐项描述组件和流水线实现。核心判断包括：

- **Memory as File**：记忆首先是用户拥有、可读写和可迁移的文件。
- **Memory from Experience**：长期记忆来自经验的提炼、修正和合并，而不是无限累积上下文。
- **Human-Agent Shared Memory**：用户和 Agent 共同读写同一份可见记忆。
- **Connected and Traceable**：长期结论可以通过链接回到来源和上下文。

算法、索引、Job、Step 和存储实现只有在帮助解释理念取舍时才进入 Concepts。

### Development 保持轻量

现代开发主要由 AI 直接阅读源码、schema 和测试完成。Development 只需要提供：

- 开发环境和最小验证命令；
- 代码目录入口；
- 兼容性与贡献要求；
- 哪些源码或 schema 是权威依据。

不为每个类、组件或扩展点编写重复的开发手册，也不维护 `generic_agent` 一类泛化教程。

### Reference 只记录稳定契约

Reference 记录 workspace、配置入口、CLI、HTTP、MCP 和文件格式的稳定语义。精确参数交给运行时帮助、Pydantic schema 和源码，避免文档复制一份容易失真的接口定义。

### Guides 只保留已验证路径

接入文档应对应真实、可验证的工作流。目前优先维护 Claude Code、QwenPaw，以及 Skill、CLI、MCP、HTTP、Python 的选择说明。没有可验证实现的框架不提前创建占位页。

## 发布文档结构

ReMe 参考 AgentScope 的版本目录和导航方式：

```text
agentscope-ai/docs/
├── reme/
│   └── 0.4.0.6/
│       ├── en/
│       └── zh/
└── images/
    └── reme/
```

每个语言版本保持三组导航：

1. **Get Started / 快速开始**：Index、Overview、Quick Start、Concepts。
2. **Integrate / 接入**：接入选择、Claude Code、QwenPaw。
3. **Reference / 查阅与参与**：Reference、Support、Contributing。

ReMe 使用项目级别的 `/reme/latest/` 和 `/reme/stable/` 别名，不影响 AgentScope 自己的 `/latest/` 与 `/stable/`。

## 变更应该写在哪里

| 变更类型 | ReMe 仓库 | 统一文档仓库 |
|---|---|---|
| 产品理念或长期设计判断 | 更新 `docs/doc.md` 或相关设计记录 | 必要时同步 Concepts |
| 用户可见的安装、配置或行为 | 源码、schema、测试；影响 README 时同步 `docs/en/`、`docs/zh/` | 更新对应版本的用户文档 |
| 内部重构或组件调整 | 以代码和测试表达 | 稳定契约未变时无需更新 |
| README 图片 | 更新 `docs/figure/` | 发布页使用时同步到 `images/reme/` |
| 新版本发布 | 更新版本号和代码 | 新建版本目录、双语导航与 ReMe 别名 |

## 质量要求

- 每个用户流程必须能够在当前版本运行和验证。
- 文档不复制能够从代码可靠获得的细节。
- 删除过期内容优先于继续叠加补丁说明。
- 中英文页面保持信息等价，不要求逐句直译。
- 发布前在统一文档仓库运行 Mintlify 严格校验。
