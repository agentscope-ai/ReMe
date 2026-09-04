---
title: 常见问题
description: ReMe 安装、服务、模型、检索、文件和插件问题的快速答案。
---

# 常见问题

## 基础文件操作需要模型 API Key 吗？

不需要。`write`、`read`、`list`、`stat`、BM25 搜索和 wikilink 遍历可以在没有模型凭据时运行。`auto_memory`、`auto_resource` 和 `auto_dream` 需要 LLM。

## 为什么配置了 Embedding Key 仍然只有 BM25？

Embedding 默认未启用。除了凭据，还必须配置 `as_embedding`、`embedding_store`，并让 `file_store.default.embedding_store` 指向该组件。参见[基础配置](./configuration.md#embedding-配置)。

## 为什么 `reme reindex` 没有发现新文件？

`reindex` 只从当前 `file_chunks` 重建 BM25 或 embedding 索引，不扫描 workspace。检查 `index_update_loop` watcher、文件目录、后缀和 `health_check`。

## 怎样使用另一个 workspace？

启动时传入稳定的绝对路径：

```bash
reme start workspace_dir=/absolute/path/to/memory
```

普通 CLI 调用会发现运行服务，不必重复 workspace 参数。

## 端口 2333 被占用怎么办？

不要停止未知监听者。选择另一个端口：

```bash
reme start service.port=8181
```

然后用 `reme find_reme` 确认发现结果。

## 为什么插件安装后仍然没有对应 Job？

安装只让 distribution 在当前 Python 环境中可见。还需要在应用配置中启用：

```bash
reme start plugins='["auto-fin"]'
```

修改安装或启用状态后需要重启已运行的服务。

## 可以直接编辑 workspace 中的 Markdown 吗？

可以。文件是事实源，watcher 会摄取修改。应保留有效 frontmatter、使用完整 workspace-relative wikilink，并避免同时由多个编辑器无条件覆盖同一文件。

## 可以把服务暴露到公网吗？

默认配置不适合直接公网暴露。服务包含写入和删除 Job，HTTP CORS 宽松，且没有通用认证层。请使用受控网络或带 TLS、认证、访问控制的反向代理，并限制 `service.jobs`。

## 怎样备份和迁移？

停止写入后备份整个 workspace。`session/`、`resource/`、`daily/` 和 `digest/` 是最重要的用户数据；`metadata/` 可以随同备份，也可以从源文件重建。详见[诊断、备份与恢复](./operations.md)。

## Studio 找不到怎么办？

基础 `reme-ai` 包不包含前端资源。安装 `reme-ai[web]` 或 `reme-ai[core]`，或通过 `service.web_static_dir` 指向构建产物。Studio 缺失不会影响 Job API。

## 当前服务到底开放了哪些能力？

运行：

```bash
reme help
reme app_config
```

静态文档描述默认配置；运行服务可能由自定义配置和插件改变。
