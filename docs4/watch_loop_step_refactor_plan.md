# Watch Loop Step 重构计划

## 背景

当前 `index_update_loop`、`resource_watch_loop`、`digest_watch_loop` 都是同一种范式：

1. 启动时扫描已有文件变化。
2. 用一组 step 处理扫描出来的变化。
3. 进入持续监听。
4. 持续监听到变化后，再用同一组 step 处理变化。

也就是说，初始化扫描和持续监听只是变化来源不同，后续更新逻辑应该共享。

现在的问题是这个范式没有被显式建模：

- `index_update_loop` 初始化和监听都走 `update_index_step`，基本一致。
- `resource_watch_loop` 初始化和监听都走 `update_catalog_step + foreach_dispatch_step`，基本一致。
- `digest_watch_loop` 初始化走 `update_catalog_step`，但监听阶段只走 `log_changes_step`，导致 live changes 不更新 `file_catalog`。
- `watch_changes_step` 同时负责监听和 dispatch 下游逻辑，职责偏重。
- `update_index_step` 和 `update_catalog_step` 内部有较多重复的变化分桶、结果收集、删除、持久化逻辑。

## 目标

重构后希望形成统一模型：

```text
change producer:
  init_changes_step     # 初始化，一次性产生 changes 并 dispatch
  watch_changes_step    # 持续监听，持续产生 changes

change handlers:
  update_index_step
  update_catalog_step
  foreach_dispatch_step
  log_changes_step
  channel_notify_step   # 后续可选
```

核心原则：

- producer 只负责产生 `context["changes"]`。
- handler 只负责消费 `context["changes"]`。
- 初始化和持续监听都通过 `BaseStep.dispatch_steps(...)` 调用 handler。
- `init_changes_step` 和 `watch_changes_step` 显式配置同一组 `dispatch_steps`，让范式直接可见。

## 配置设计

不新增 job 级 `change_steps`。每个 producer step 自己声明 `dispatch_steps`，初始化 producer 和持续监听 producer 配同一组 handler。

配置精简约定：

- `init_changes_step.recursive` 和 `watch_changes_step.recursive` 默认就是 `true`，配置中不再显式写。
- `update_index_step` / `update_catalog_step` 的 `persist` 语义统一为默认 `true`，配置中不再显式写。
- 只有当某个 loop 需要关闭递归或关闭持久化时，才显式写 `recursive: false` / `persist: false`。

### index_update_loop

```yaml
index_update_loop:
  backend: background
  watch_dirs: [daily_dir, digest_dir, resource_dir]
  watch_suffixes: [md, jsonl]
  steps:
    - backend: init_changes_step
      store: file_store
      dispatch_steps: [update_index_step]
    - backend: watch_changes_step
      dispatch_steps: [update_index_step]
```

### resource_watch_loop

```yaml
resource_watch_loop:
  backend: background
  watch_dirs: [resource_dir]
  watch_suffixes: [md, txt, json, jsonl, csv, yaml, html]
  dispatch_job: auto_resource
  steps:
    - backend: init_changes_step
      store: file_catalog
      dispatch_steps: [update_catalog_step, foreach_dispatch_step]
    - backend: watch_changes_step
      dispatch_steps: [update_catalog_step, foreach_dispatch_step]
```

### digest_watch_loop

```yaml
digest_watch_loop:
  backend: background
  watch_dirs: [daily_dir, digest_dir]
  watch_suffixes: [md]
  steps:
    - backend: init_changes_step
      store: file_catalog
      dispatch_steps: [update_catalog_step, log_changes_step]
    - backend: watch_changes_step
      dispatch_steps: [update_catalog_step, log_changes_step]
```

这样三个 loop 都统一成：

```text
startup:
  scan changes
  dispatch handlers

runtime:
  watch changes
  dispatch same handlers
```

## Step 拆分

### 1. BaseStep dispatch 能力

把 dispatch 能力沉到 `BaseStep`，所有 producer step 共享：

- `normalize_dispatch_steps(dispatch_step, dispatch_steps)`
- `dispatch_steps(dispatch_steps, **kwargs)`

这样 `init_changes_step` 和 `watch_changes_step` 都不需要各自实现 registry 查询、step 实例化和 context 透传。

`dispatch_steps` 支持两种形式：

```yaml
dispatch_steps: [update_catalog_step, log_changes_step]
```

也支持给单个 handler 传参数：

```yaml
dispatch_steps:
  - backend: update_catalog_step
  - backend: some_step
    option: value
```

### 2. init_changes_step

新增 `init_changes_step`，替代现有两个初始化扫描 step：

- `scan_store_changes_step`
- `scan_catalog_changes_step`

参数：

```yaml
store: file_catalog | file_store
dispatch_steps: [...]
```

职责：

- 根据 `watch_dirs` / `watch_suffixes` 收集磁盘文件。
- 根据 `store` 和目标状态源比较。
- 生成统一格式的 `context["changes"]`。
- 如果有变化，调用 `BaseStep.dispatch_steps(...)` 执行 handler。

输出格式：

```python
[
    {"change": "added", "path": "/abs/path/to/file.md"},
    {"change": "modified", "path": "/abs/path/to/file.md"},
    {"change": "deleted", "path": "/abs/path/to/file.md"},
]
```

### 3. watch_changes_step

保留监听职责，弱化业务 dispatch 职责。

职责：

- 根据 `watch_dirs` / `watch_suffixes` 建立文件监听。
- 对每个 debounced batch 生成同样格式的 `changes`。
- 调用 `BaseStep.dispatch_steps(...)` 执行 handler。

### 4. update_index_step / update_catalog_step

第二阶段再精简。

它们现在重复逻辑包括：

- 解析 `added` / `modified` / `deleted`。
- 判断文件是否存在。
- 收集 per-path result。
- 删除旧记录。
- upsert 新记录。
- persist。
- 写 response。

建议抽内部基类，例如：

```python
class ChangeApplyStep(BaseStep):
    async def parse_added_or_modified(self, path): ...
    async def upsert_items(self, items): ...
    async def delete_paths(self, rel_paths): ...
    async def dump_target(self): ...
```

然后：

- `UpdateCatalogStep` 只实现 `stat -> FileNode`，写 `file_catalog`。
- `UpdateIndexStep` 只实现 `chunk_file -> FileNode + chunks`，写 `file_store`。

这一步可以在 `init_changes_step` 落地后做，降低一次性改动风险。

## 实施顺序

当前落地状态：

- Phase 1 已完成：`BaseStep.dispatch_steps(...)`、`init_changes_step`、默认持久化、默认配置精简已落地。
- Phase 2 已完成：`digest_watch_loop` 的初始化和监听都执行 `update_catalog_step + log_changes_step`。
- 兼容旧 backend 不保留：`scan_store_changes_step` / `scan_catalog_changes_step` 已删除。
- `reindex` 已改为 `clear_store_step + init_changes_step(store=file_store, dispatch_steps=[update_index_step])`。
- Phase 3 已完成一层：`update_catalog_step` 和 `update_index_step` 已合并到 `update_changes.py`，
  并抽出 `ChangeApplyStep` 复用 added/modified/deleted、upsert/delete/persist 模板逻辑。

### Phase 1：统一范式

1. 把 dispatch 能力沉到 `BaseStep`。
2. 新增 `init_changes_step`。
3. 将 `update_index_step` 和 `update_catalog_step` 的 `persist` 默认值统一为 `true`。
4. 修改 `default.yaml` 里的三个 loop：
   - 初始化阶段统一使用 `init_changes_step`。
   - `init_changes_step` 和 `watch_changes_step` 配置相同的 `dispatch_steps`。
5. `watch_changes_step` 保留 `dispatch_step` 到 `dispatch_steps` 的轻量兼容。
6. 验证三个 loop 的启动扫描和 live watch 都会执行同一条 handler pipeline。

### Phase 2：修正 digest_watch_loop 语义

`digest_watch_loop` 的 live changes 应该更新 `file_catalog`，因此初始化和监听阶段都应配置同一组 `dispatch_steps`：

```yaml
dispatch_steps: [update_catalog_step, log_changes_step]
```

如果后续要通知 channel，可以追加：

```yaml
  - backend: channel_notify_step
```

### Phase 3：精简 update 类 step

1. 抽 `ChangeApplyStep` 基类或 helper 函数。
2. 让 `update_index_step` 和 `update_catalog_step` 只保留各自差异逻辑。
3. 保持外部行为不变：
   - 输入仍然是 `context["changes"]`。
   - 输出仍然写 `response.answer` 和 `response.success`。
   - `persist` 语义不变。

## 验证点

最少需要覆盖这些场景：

- `index_update_loop` 启动扫描新增文件，会更新 `file_store`。
- `index_update_loop` live 新增/修改/删除文件，会更新 `file_store`。
- `resource_watch_loop` 启动扫描新增文件，会更新 `file_catalog` 并触发 `auto_resource`。
- `resource_watch_loop` live 新增文件，会更新 `file_catalog` 并触发 `auto_resource`。
- `digest_watch_loop` 启动扫描新增/修改/删除文件，会更新 `file_catalog`。
- `digest_watch_loop` live 新增/修改/删除文件，也会更新 `file_catalog`。
- `changes` 为空时，`init_changes_step` 不应执行 handler，也不应报错。

## 预期收益

- 三个 background loop 的结构统一。
- 初始化扫描和持续监听的处理逻辑完全复用。
- `digest_watch_loop` 不再出现启动和 live 语义不一致。
- `watch_changes_step` 和 `init_changes_step` 共享 `BaseStep` dispatch 能力。
- 后续新增日志、channel notification、auto dream 等 handler 时，初始化和监听两处使用同一组 `dispatch_steps`。
