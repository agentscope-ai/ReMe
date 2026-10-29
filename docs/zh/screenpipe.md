# 导入已审核的 Screenpipe 笔记

[Screenpipe](https://github.com/screenpipe/screenpipe) 记录桌面活动和会议上下文。
可以把从这些记录中选出并审核过的笔记作为 ReMe 资源，生成每日记忆卡片，供后续 Agent 会话检索。
本指南使用现有的资源处理流程，手动导入笔记。

## 准备来源笔记

在 Screenpipe 中选择一次已结束的会议或工作片段。审核并修正保存的笔记，然后只把希望 ReMe 保留的信息
复制到 UTF-8 Markdown 文件中。先将文件保存在 ReMe 工作区之外，并包含：

- 标题，以及带时区的记录时间。
- Screenpipe 会议 ID，或可以在本机查找的其他来源标识。
- 审核后的笔记，以及尚未确认的信息。

以下为虚构示例：

```markdown
# 示例项目交接

来源：my-work-laptop 上的 Screenpipe 会议 42
记录时间：2026-01-15T10:00:00Z
已审核：是

团队选择用 CSV 导出进行交接。交付日期尚未确定。
```

保持来源设备标识稳定：不同安装中的会议 ID 可能相同。描述工作流程的笔记本身不能证明每一步都已成功执行。

## 添加到工作区

完成[快速开始](./quick_start.md)，确认正在运行的 ReMe 服务所使用的工作区。
把审核后的文件复制到该工作区，例如：

```text
resource/2026-01-15/screenpipe-my-work-laptop-meeting-42.md
```

目录名使用会议日期。不要覆盖无关文件。审核完成后再复制：已启用的资源监听器可能在文件出现后立即处理它。
处理过程可能将笔记发送给 ReMe 配置的模型提供方。

如果资源监听器未启用，可以显式指定同一个工作区并调用现有 Job：

```bash
reme auto_resource workspace_dir=/absolute/path/to/your/workspace changes='[{"path":"resource/2026-01-15/screenpipe-my-work-laptop-meeting-42.md","change":"added"}]'
```

将示例路径替换为实际资源路径。首次导入时选择监听器或显式调用之一，不要同时使用两种方式。

## 确认后再使用记忆

检查 `daily/2026-01-15/` 下生成的卡片，确认 `source_resource` 指向复制的 Markdown 文件。
将卡片与原始笔记对照。随后通过 ReMe 的[搜索和读取接口](./memory_search.md)，在另一会话中检索交接决定，
并保留来源路径。文件复制成功不等于解释和索引成功；重试失败的 Job 前先检查错误。

## 修正和删除

修正笔记时保留资源路径。监听器可以处理修改；如果监听器未启用，则使用 `change="modified"` 调用
`auto_resource`。现有资源契约通过精确的 `source_resource` 链接更新对应卡片。

删除 Screenpipe 中的原始记录不会删除导入的副本。要从 ReMe 删除该资源，可以删除选定的资源文件并让已启用的
监听器处理删除事件，或显式提交该路径并设置 `change="deleted"`。另外检查后续生成的 digest 知识；
删除来源卡片不代表所有衍生知识都已清除。

处理、重试和删除语义见 [Auto Resource](./auto_resource.md)。
