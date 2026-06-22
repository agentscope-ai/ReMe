# Auto Link

当前实现中，`auto_link` 不是一个单独注册的 Job，而是 `auto_dream` 的 Integrate 阶段能力：`dream_integrate_step` 在把 memory
unit 写入 `digest/` 时，同时完成 digest 节点召回、去重判断、来源链接和相关节点 wikilink 织入。

完整 dream 流程见 [Auto Dream](./auto_dream.md)。通用 wikilink、frontmatter 和 workspace-relative 路径语义见
[Memory as File](./memory_as_file.md)。面向问答的检索能力见 [Memory Search](./memory_search.md)。

## 所在位置

`auto_dream` 的默认流程如下：

```yaml
auto_dream:
  steps:
    - dream_extract_step
    - dream_integrate_step   # auto_link 的实际发生位置
    - dream_topics_step
    - dream_finish_step
```

Integrate 阶段对每个 unit 独立运行。一个 unit 只落到一个目标 digest 节点，但这个目标节点可以链接多个来源和多个相关 digest
节点。

## 目标

`auto_link` 解决的是写入时的图谱质量问题：

| 问题           | 处理方式                                               |
|--------------|----------------------------------------------------|
| 已有相同记忆       | 召回后更新旧节点，而不是重复创建。                                  |
| 新旧材料有关联      | 在正文中写入 workspace-relative wikilink。                    |
| digest 与来源断开 | 用 `derived_from:: [[...]]` 指回 daily/resource 原始材料。 |
| 节点只有孤立正文     | 在 CREATE 和 UPDATE 时都补充相关 digest 节点链接。              |

## 工具链

`dream_integrate_step` 暴露给 Agent 的工具是：

```text
node_search
read
frontmatter_read
write
edit
frontmatter_update
```

其中 `node_search` 是为 dream 集成设计的 digest-only 节点召回。它返回 digest 节点的 `path`、front matter 中的 `name`、
`description` 等节点级信号，不展开正文，也不做普通 search 的 link expansion。

`read` 和 `frontmatter_read` 只用于可能相关的候选节点，避免把召回结果全部展开成大上下文。

## 链接流程

### 1. 召回候选节点

Agent 先用 unit 的触发条件、动词、名词、同义词和可能的 failure modes 调用 `node_search`。默认建议用较宽召回，例如
`limit=20-30`，因为这一步同时服务去重和链接发现。

召回结果会被内部分成三类：

| 分类                 | 含义                          | 后续动作           |
|--------------------|-----------------------------|----------------|
| `same_abstraction` | 触发条件或抽象本质相同，内容实质重叠。         | 作为 UPDATE 目标。  |
| `related`          | 相邻流程、前置条件、失败模式、概念、偏好或上下游知识。 | 写入正文 wikilink。 |
| `unrelated`        | 只是表面相似或无关。                  | 忽略。            |

### 2. 选择写入动作

每个 unit 必须选择一个动作：

| 动作            | 链接含义                                                |
|---------------|-----------------------------------------------------|
| `CREATE`      | 写入新的 `digest/<bucket>/<slug>.md`，并在新正文里加入来源和相关节点链接。 |
| `CORROBORATE` | 同一抽象再次出现，追加新的 `derived_from:: [[...]]`，必要时强化描述。     |
| `REFINE`      | 新材料扩展了旧节点，把补充内容插入合适段落，并保留旧链接。                       |
| `CORRECT`     | 新材料修正旧节点，用来源链接标出修正依据。                               |

UPDATE 必须尽量只增不删：不要删除已有 wikilink 或 `derived_from`。这是为了让后续图谱索引和检索不会丢边。

### 3. 写来源边

来源边使用 markdown wikilink：

```markdown
derived_from:: [[daily/2026-06-20/session.md]]
derived_from:: [[resource/2026-06-20/paper.md]]
```

这些边表示 digest 节点的证据来源。纯文本描述不算来源边，因为只有 wikilink 能被 file graph 稳定解析。更完整的 wikilink
解析规则见 [Memory as File](./memory_as_file.md#wikilink)。

### 4. 写 digest 关联边

digest 之间的关联也使用完整 workspace-relative 路径：

```markdown
relates_to:: [[digest/wiki/hybrid-search.md]]
depends_on:: [[digest/procedure/rebuild-index.md]]
blocks_on:: [[digest/personal/team-review-preference.md]]
```

谓词是开放的，常见写法包括 `relates_to::`、`depends_on::`、`blocks_on::`。谓词在括号外，目标路径在 `[[...]]` 内，并且应包含
`.md` 后缀。

## Bucket 差异

`auto_link` 的规则会随 unit bucket 调整写入形态：

| Bucket      | 写入重点                                        |
|-------------|---------------------------------------------|
| `procedure` | 写成 runbook：触发条件、步骤、输入、失败模式。链接前置流程、子步骤、相关偏好。 |
| `personal`  | 写用户、团队、项目特定事实或偏好。链接相关项目、习惯、决策背景。            |
| `wiki`      | 写通用知识、原则、观察、决策先例。链接概念、方法、相邻知识。              |

无论 bucket 是什么，都要保留来源边，并尽量把召回到的相关 digest 节点织入正文。

## 与 search 的关系

`auto_link` 使用的是 `node_search`，不是面向问答的 `search`。

| 能力            | 用途                                        |
|---------------|-------------------------------------------|
| `search`      | 面向外部问答，返回 chunk，并可展开上下游 link context。     |
| `node_search` | 面向 dream 集成，只召回 digest 节点级摘要，用来判断去重和相关链接。 |

这个边界很重要：Integrate 阶段需要的是“是否已有相同抽象，以及应该链接哪些节点”，而不是直接把大量正文片段塞进上下文。
[Memory Search](./memory_search.md) 负责面向用户问题的 chunk 召回、RRF 融合和链接展开。

## 失败和重试

如果某个 unit 整合失败，`dream_integrate_step` 会记录 `failed_units` 和 `failed_paths`。`dream_finish_step` 不会
checkpoint 这些来源路径，因此下一次 `auto_dream` 仍会重新处理它们。

这保证了 auto_link 的写入具有可重试性：失败不会把输入标成已完成，也不会静默丢失应该建立的 digest 边。
