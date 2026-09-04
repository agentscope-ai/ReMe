---
title: 诊断、备份与恢复
description: ReMe 服务健康检查、日志、索引维护、备份迁移和常见恢复流程。
---

# 诊断、备份与恢复

ReMe 的恢复原则是：保护 workspace 中的用户文件，通过源文件重建索引、图谱和 catalog。不要为了修复索引而删除或改写用户记忆。

## 快速诊断

依次执行：

```bash
reme find_reme
reme version
reme health_check
reme status
reme app_config
```

- `find_reme`：服务是否存在，以及实际 host、port、PID；
- `version`：CLI 能否成功访问服务；
- `health_check`：组件健康状态；
- `status`：状态组件的内存估算和进程 RSS；
- `app_config`：隐藏密钥后的实际生效配置。

## 日志

`log_to_console` 和 `log_to_file` 控制日志目标。排查启动失败时先查看第一条异常，而不是后续客户端连接错误。

常见类别：

| 现象 | 优先检查 |
|---|---|
| CLI 找不到服务 | `reme find_reme`、启动目录、端口和进程状态 |
| 自动记忆失败 | LLM backend、model、API key、base URL |
| 只有 BM25 结果 | embedding 组件是否真正接入 `file_store` |
| 新文件没有进入搜索 | 文件所在目录、后缀、watcher 和 `health_check` |
| Studio 空白但 API 正常 | web extra、静态构建路径和浏览器控制台 |
| 插件安装后不可用 | 当前 Python 解释器、`plugins` 配置、服务重启 |

## 索引维护

```bash
reme reindex scope=all
reme reindex scope=bm25
reme reindex scope=embedding
```

`reindex` 从当前 `file_chunks` 重建 BM25 和/或 embedding 派生索引。它不会扫描 workspace、重新分块或重建 wikilink 图谱。

如果问题发生在文件摄取阶段，应先确认后台 watcher 正常运行；不能把 `reindex` 当成通用“重新扫描”命令。

Daily 索引页可单独重建：

```bash
reme daily_reindex date=2026-09-04
```

## 备份

停止写入或停止服务后，优先备份整个 workspace。最重要的目录是：

- `session/`：原始对话来源；
- `resource/`：外部资料；
- `daily/`：每日记忆；
- `digest/`：长期记忆。

`metadata/` 包含索引、图谱和 file catalog。一起备份可以加速恢复，但它不是唯一事实源。

不要只备份进程目录下偶然生成的 `.reme/`；生产使用应明确设置稳定的绝对 `workspace_dir`。

## 迁移 workspace

1. 停止旧服务，避免迁移期间继续写入。
2. 复制完整 workspace，并保留文件时间信息。
3. 使用新的绝对路径启动：

```bash
reme start workspace_dir=/new/location/reme-memory
```

4. 运行 `health_check`、`status` 和一次代表性 `search`。
5. 如果 embedding 模型或维度发生变化，再重建 embedding 索引。

普通 Markdown 和资源文件可以使用版本控制或同步工具；包含敏感对话的 workspace 不应推送到公开仓库。

## 从派生状态故障恢复

在确认备份可用前，不要删除任何内容。恢复顺序应为：

1. 保留 `session/`、`resource/`、`daily/` 和 `digest/`；
2. 记录当前配置和组件 backend；
3. 确认故障只发生在 `metadata/`；
4. 将有问题的派生状态移到隔离备份位置；
5. 用同一配置启动 ReMe，让 watcher 从源文件重建；
6. 验证搜索、图谱和 Daily 索引。

具体 metadata 文件属于实现细节，不应在自动化脚本里依赖其内部格式。

## 并发编辑

Studio 或其他编辑器保存完整文件时，应使用 `stat` 返回的 mtime 作为 `save.expected_mtime`。如果文件在打开后被外部修改，保存会失败，从而避免静默覆盖。

文件 Job 会校验 workspace containment，并对同一路径加锁。不要绕过这些 Job 直接向不受控制的绝对路径写入。
