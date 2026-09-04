---
title: CLI Reference
description: ReMe command syntax, service invocation, configuration overrides, and plugin commands.
---

# CLI Reference

The basic syntax is:

```text
reme ACTION key=value ...
```

## Start an Application

```bash
reme start
reme start config=demo
reme start workspace_dir=/data/reme service.port=8181
reme start job=search query="keywords" limit=5
```

`start job=<name>` runs one Job through a one-shot service; plain `start` runs the configured Service.

## Call Jobs

Once a service is running, each action name is a Job name:

```bash
reme help
reme health_check
reme search query="project decision" limit=10
reme read path=digest/wiki/project.md start_line=1 end_line=80
```

Use JSON for structured values:

```bash
reme auto_memory \
  session_id=example \
  messages='[{"role":"user","content":"Remember this preference"}]'
```

Client-selection arguments—`backend`, `transport`, `host`, `port`, `timeout`, `command`, `args`, and `show_metadata`—configure the client and never leak into the Job payload.

## Configuration overrides

```bash
reme start \
  config=/path/to/custom.yaml \
  service.port=8181 \
  service.web_enabled=false \
  plugins='["auto-fin"]'
```

Leading `-` or `--` is optional. Use dots for nested keys and JSON for arrays and objects.

## Service discovery

```bash
reme find_reme
```

This reports a discovered service but never starts or replaces a process.

## Plugin commands

Package management runs locally rather than through HTTP or MCP:

```bash
reme plugins list
reme plugins show auto-fin
reme plugins validate auto-fin
reme plugins install reme-auto-fin
reme plugins uninstall auto-fin
```

See [Plugin Management](../plugin_management.md) for the complete workflow.

## Discover active capabilities

The [Job API Reference](./jobs.md) describes the default configuration. Plugins and custom YAML may change the running service, so automation should prefer:

```bash
reme help
reme app_config
```
