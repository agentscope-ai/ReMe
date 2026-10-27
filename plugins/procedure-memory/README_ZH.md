# Procedure Memory

ReMe 的可选经验记忆插件。将本机轨迹提炼为可复用经验，原文件按原字节保存，每条经验保存为一个 JSON 文件，并建立两个 vector + BM25 混合索引。现有事实记忆行为不变。

## 安装和配置

在 ReMe 仓库中执行：

```bash
pip install -e '.[core,qwen-tokenizer]' -e plugins/procedure-memory
```

通过 `plugins='["procedure-memory"]'` 启用。提炼使用通常的 `LLM_*` 配置，向量使用 `EMBEDDING_BACKEND`、`EMBEDDING_MODEL_NAME`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`；插件为自己的索引显式开启 embedding。维度可通过 `components.as_embedding.procedure.dimensions` 调整。

`PROCEDURE_TOKENIZER_SOURCE` 指定 tokenizer 名称或本机文件/目录，默认 `Qwen/Qwen3.8-27B`。请选择与推理模型匹配的 tokenizer。精确 tokenizer 不可用时返回可恢复失败，不回退为估算。首次使用远端名称可能下载 tokenizer；本机路径支持离线运行。

## 本地导入与提炼（one-shot CLI）

一个文件代表一条轨迹，一次处理一个文件；JSONL 的多条记录仍属于同一轨迹。本机导入仅通过本地 CLI 或同进程 `Application` 调用，不作为 HTTP/MCP 上传接口。运行单次导入或重建前，停止访问同一 workspace 的其他 ReMe 进程。

```bash
reme start job=procedure_auto_memory plugins='["procedure-memory"]' \
  file=/absolute/path/run.jsonl filename=run.jsonl request_id=run-001
```

`file`、`filename` 必填；`memory_hint` 可指定关注点，`source_id` 可提供调用方标签。稳定的 `request_id` 用于重试和续跑，复用原始副本及检查点。续跑仍传最初的 `file` 路径和 `filename`；工作区副本已发布后，原始文件可以删除或移动，恢复时会校验副本与导入记录的校验和。原始文件仍存在但内容改变时会报错；副本尚未保存成功时仍需要原始文件。不传 request ID 时每次都是新导入，响应返回生成的 request ID。重复 ID 的输入路径、文件名或不兼容的处理配置变化会明确报错。context/reserve 和 SDK 轮数预算可调整后续跑，不改变原有分段。

第一版支持纯文本（含无扩展名文件）、Markdown、JSON、JSONL、日志；JSON/JSONL 不丢弃字段。不支持或格式错误时保留原文件并返回未完成状态。原文件保持格式与字节，不转为 txt；默认文件大小上限为 20 MiB，可配置 `components.procedure_memory.default.max_file_bytes`。

长轨迹逐段处理：前一滚动摘要 + 当前段 → agent 搜索、读取、创建或修改经验 → 更新摘要 → 保存检查点。可产生零条或多条经验，无经验数目限制。三个文本字段均可改写，来源由程序保留合并。每次写后立即同步两个索引；字段超限要求缩写，不静默截断。

## 服务检索

本地导入完成后，采用两阶段服务调用：

```bash
# 终端一
reme start plugins='["procedure-memory"]'

# 终端二
reme procedure_search query='修改配置后索引失效' index=when_to_use
reme procedure_search query='重建前先检查日志' index=when_to_use_content limit=5
reme procedure_read path=procedures/rebuild-search-after-config-change.json
```

两个视图分别索引 `when-to-use`、`when-to-use + "\n\n" + content`，每条经验每视图只有一个完整记录，不分块。两个视图使用独立的 BM25 和 graph，共享 embedding cache，通过加权 RRF 融合排序，默认 vector 权重 0.7。两者均返回完整最新经验、相对路径、内容版本和分数，重复检索不会隐藏已见经验。

auto-memory 获得 `procedure_search`、`procedure_read`、`procedure_create`、`procedure_update` 四个 Job 工具。写入能力和真实来源由运行时注入，仅提炼期间可用；修改须提供搜索/读取返回的 `expected_version`。可通过 `procedure_read kind=trajectory path=session/trajectories/run.jsonl offset=0 length=8192` 回读原文；offset/length 是解码后的字符位置，同时受 token 上限约束。

## 存储契约

```text
<workspace>/
  session/trajectories/run.jsonl
  session/trajectories/run-2.jsonl
  procedures/rebuild-search-after-config-change.json
  metadata/procedure/default/{imports,operations,processing}/
  metadata/{file_store,file_graph,keyword_index,embedding_store}/...
```

`session` 跟随核心 `session_dir`；插件组件中可配置 `procedure_dir`、`trajectory_subdir`。目标目录及工具读取均限制在 workspace 和对应目录中，只有显式导入的本机源路径可在 workspace 外。

经验 JSON 仅有四字段：

```json
{
  "description": "配置变化后重建检索索引",
  "when-to-use": "修改检索配置后，结果仍然过期时",
  "content": "重建受影响索引，然后使用已知查询验证结果。",
  "sources": ["session/trajectories/run.jsonl"]
}
```

经验由 agent 以几个有意义的英文单词命名，使用小写和连字符并以 `.json` 结尾，相对路径就是稳定标识。经验重名须读取后修改或另取语义名称，修改不重命名。原始轨迹重名则在最后一个扩展名前依次加 `-2`、`-3`；中断导入预留的名称也不会被其他请求占用。独立导入不按内容去重。

`sources` 仅含实际保存的原始文件相对路径，自动累积去重；无 `source.json`、公开 `trace_id` 或独立 experience ID。

默认 token 上限：轨迹段 98,304（96K）、滚动摘要目标 8,192（8K）、description 128、when-to-use 512、content 2,048。摘要和三个经验字段校验均在内部允许 10% 余量，默认实际接受上限为摘要 9,011、description 140、when-to-use 563、content 2,252 token；模型只收到配置的目标值，不告知余量。分段和上下文预算仍严格限制。初始输入、摘要和工具 schema 总预算 262,144，预留 4,096，需配置上下文容量足够的模型。通过 `components.procedure_memory.default.token_limits` 调整。SDK 运行轮数属于技术预算，未完成时可续跑，不作为经验条数限制。已有检查点续跑时须使用原分段配置。

## 恢复与维护

写入前记录操作意图、目标路径和内容。文件已落盘但索引失败时，写工具返回失败的 JSON 响应，包含 `status=index_pending`、实际路径及版本。临时失败可重试同一操作；内容被永久拒绝时，可由提炼 agent 使用 `procedure_read` 读取当前版本，再以新的 `operation_key` 和当前 `expected_version` 调用 `procedure_update` 原地修正。修正意图先持久化，再取代旧失败操作；修正中断或重启都不会恢复旧内容，不需要换名重建。

失败经验的 JSON 保留，并暂时从**两个**搜索视图排除；启动及其他轨迹仍可继续。检索返回可用经验，同时明确设置 `index_complete=false`，通过 `pending_indexes` 列出排除的路径、版本与错误，避免静默返回过期记录。修正并完成同步后重新进入两个视图。当前段若仍有待同步写入，不推进检查点；续跑会再次调用 agent 进行修正。若底层索引故障导致无法安全排除记录，仍明确报错。

检查点保存段位置、摘要、配置指纹、提炼输入及最终响应。指纹包含分段算法版本；旧分段算法创建的未完成任务须先使用原插件版本完成，再升级。不兼容检查点明确报错，不改动原文件或进度。分段根据实测 token 密度预测有界候选，目标为 token 上限的约 95%，超限时减半重试。每个输出段均经过精确计数；优先选择候选后半区的完整行边界，必要时拆分记录。文本无损，但不保证填满预算。续跑将已有操作记录交给 agent，以读取已写经验后继续；若最终响应已保存但段位置未推进，复用响应。启动时恢复未完成操作，并从经验 JSON 重建索引。维护操作不改写或删除原始轨迹和经验。

停止同 workspace 的其他写入进程后，可显式重建：

```bash
reme start job=procedure_reindex plugins='["procedure-memory"]'
```

重建会报告非法 JSON 文件。第一版只支持单进程，进程内串行提炼文件，不监听用户手动修改；不索引轨迹和摘要，也不自动删除、重命名、合并或拆分经验。

## 验证

```bash
cd plugins/procedure-memory
pytest -q
```

测试使用临时 workspace、真实 ReMe Job/FileStore/BM25，以及模拟 agent、embedding、token 计数器，无需外部模型或网络。公开 schema 见 [schema.py](src/reme_procedure_memory/schema.py)，默认配置见 [plugin.yaml](src/reme_procedure_memory/plugin.yaml)。
