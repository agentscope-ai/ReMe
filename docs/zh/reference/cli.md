---
title: CLI 参考
description: ReMe 命令行语法、服务调用、配置覆盖与插件命令。
---

# CLI 参考

ReMe 的基本语法是：

```text
reme ACTION key=value ...
```

## 启动应用

```bash
reme start
reme start config=demo
reme start workspace_dir=/data/reme service.port=8181
reme start job=search query="关键词" limit=5
```

`start job=<name>` 运行一次性 Job；普通 `start` 启动配置中的 Service。

## 调用 Job

服务运行后，Action 名就是 Job 名：

```bash
reme help
reme health_check
reme search query="项目决策" limit=10
reme read path=digest/wiki/project.md start_line=1 end_line=80
```

复杂值使用 JSON：

```bash
reme auto_memory \
  session_id=example \
  messages='[{"role":"user","content":"记住这条偏好"}]'
```

服务选择参数 `backend`、`transport`、`host`、`port`、`timeout`、`command`、`args` 和 `show_metadata` 只用于构造客户端，不会泄漏到 Job 参数。

## 配置覆盖

```bash
reme start \
  config=/path/to/custom.yaml \
  service.port=8181 \
  service.web_enabled=false \
  plugins='["auto-fin"]'
```

参数前的 `-` 或 `--` 可以省略。嵌套键使用点号，数组和对象使用 JSON。

## 服务发现

```bash
reme find_reme
```

成功时输出可复用的服务信息；找不到服务时不会自动启动新进程。

## 插件命令

插件包管理是本地命令，不经过 HTTP 或 MCP：

```bash
reme plugins list
reme plugins show auto-fin
reme plugins validate auto-fin
reme plugins install reme-auto-fin
reme plugins uninstall auto-fin
```

完整说明见[插件管理](../plugin_management.md)。

## 发现当前能力

默认 Job 参数见 [Job API 参考](./jobs.md)。运行配置可能由插件和自定义 YAML 改变，因此自动化程序应优先调用：

```bash
reme help
reme app_config
```
