---
title: Docker 部署
description: 使用 Docker 运行 ReMe 和 Studio，并将持久记忆保存在用户拥有的工作区中。
---

# Docker 部署

镜像包含 ReMe 的 `core`、HEIF 图片依赖和 Studio 静态前端。一个 HTTP 进程同时提供 API、根路径 `/` 上的 Studio 和
`/mcp` 上的 MCP。默认配置关闭 embedding；基础文件操作和 BM25 检索不需要模型凭证。

## 使用 Compose 构建并启动

需要 Docker Engine 或 Docker Desktop，以及 **2.24.0 或更新版本**的 Compose。在仓库根目录运行：

```bash
mkdir -p .reme
docker compose up --build -d
docker compose logs -f reme
```

打开 <http://127.0.0.1:2333>。Compose 默认只开放宿主机回环地址，并将 `./.reme` 挂载到容器的 `/data`。
源码目录和 Studio 资源不会覆盖镜像内已经安装的应用。

在 Linux 上，如果工作区所有者的 UID/GID 不是 1000，请在启动前设置运行用户：

```bash
export REME_UID=$(id -u)
export REME_GID=$(id -g)
docker compose up --build -d
```

需要模型驱动的记忆演化时，如果尚无 `.env`，将 `deploy/docker/example.env` 复制为 `.env`，再填写模型凭证。
Compose 在运行时注入这个可选文件；Docker 构建排除 `.env` 文件。Compose 的 `--env-file` 用于配置变量替换，
`REME_ENV_FILE` 则选择实际注入容器的环境文件。

## 使用发布镜像

Docker 工作流在 main 分支检查成功后发布 `ghcr.io/agentscope-ai/reme:main`。稳定 GitHub Release 发布对应的包版本和
`latest`；预发布版本不会更新 `latest`。Linux amd64 和 arm64 镜像分别通过验证后才发布共同的标签。
镜像发布从仓库启用该工作流后开始生效。

```bash
mkdir -p "$HOME/.reme"
docker run -d --name reme \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:2333:2333 \
  --mount "type=bind,source=$HOME/.reme,target=/data" \
  --restart unless-stopped \
  ghcr.io/agentscope-ai/reme:main
```

需要模型凭证时，在镜像名之前添加 `--env-file /path/to/model.env`。需要固定部署版本时，将 `main` 替换为发布版本或
镜像 digest。使用 Compose 拉取镜像时，设置 `REME_IMAGE`，再运行 `docker compose pull` 和
`docker compose up -d --no-build`。

## 路径、配置和端口

镜像默认以 UID/GID 1000 运行。`/data` 保存完整工作区，包括源对话、资料、daily、digest 和可重建的 metadata。
需要自己创建宿主机目录，并确保运行用户具有写权限。仅挂载 `metadata/` 不能保存源记忆。
自定义配置中的路径指向容器文件系统；额外路径需要额外挂载。

| 配置项 | 含义 |
|---|---|
| `REME_WORKSPACE_DIR` | 容器工作区；镜像默认 `/data`，Compose 固定为 `/data` |
| `REME_CONFIG` | 已有配置名称或挂载的 YAML/JSON 路径；未设置时使用内置默认配置 |
| `REME_HOST` | HTTP 监听地址；镜像和 Compose 使用 `0.0.0.0` |
| `REME_PORT` | 容器端口覆盖；Compose 默认 `2333` |
| `REME_TIMEZONE` | 可选的应用时区覆盖；未设置时沿用应用默认值 |
| `REME_DATA_DIR` | Compose 的宿主机工作区目录；默认 `./.reme` |
| `REME_PUBLISHED_PORT` | Compose 的宿主机端口；默认 `2333`，与容器端口独立 |
| `REME_BIND_ADDRESS` | Compose 的宿主机监听地址；默认 `127.0.0.1` |
| `REME_UID`、`REME_GID` | Compose 的运行用户；均默认 `1000` |
| `REME_ENV_FILE` | 可选的 Compose 运行时环境文件；默认 `.env` |

显式的 `start key=value` 参数优先于容器环境配置；容器环境配置再覆盖文件中的对应键。其他键继续采用 ReMe 的正常
深合并规则。镜像默认关闭文件日志，使用容器日志查看运行情况。显式传入 `log_to_file=true` 可启用 `/app/logs` 下的文件
日志，需要另外挂载才能持久保存。临时 home `/tmp/reme-home` 和探针地址文件均为可丢弃数据，不承担工作区存储职责。

需要自定义完整的 Job/Component 配置时，将 `reme/config/default.yaml` 复制为 `reme.yaml` 并修改，在 `.env` 中设置
`REME_CONFIG=/etc/reme/config.yaml`，再添加 `compose.override.yaml`：

```yaml
services:
  reme:
    volumes:
      - ./reme.yaml:/etc/reme/config.yaml:ro
```

保留启用的 `health_check` Job；如果配置 `service.jobs` 白名单，也要包含它。镜像探针使用 HTTP；将服务覆盖为 CLI 或
MCP stdio 时，请通过 `--no-healthcheck` 关闭 Docker 探针，Compose 中则使用 `healthcheck: {disable: true}`。

无需修改镜像即可覆盖启动配置：

```bash
docker run --rm -p 127.0.0.1:2444:2444 \
  --mount "type=bind,source=$HOME/.reme,target=/data" \
  reme:local start service.port=2444 timezone=UTC
docker compose exec reme reme health_check
docker compose exec reme reme status
```

其他命令直接执行，例如 `reme start job=version` 可运行一次性 Job，`python` 可用于诊断。
宿主机 CLI 应明确指定发布地址，例如 `reme health_check host=127.0.0.1 port=2444`；宿主机的进程发现无法还原容器内
启动参数。Agent 集成应连接发布的 HTTP 或 MCP 地址，避免对同一工作区再启动第二个本机 ReMe。

## 网络和可选工具

容器内的 `127.0.0.1` 指向容器自身。宿主机上的模型服务需要使用可访问的宿主机地址，例如 Docker Desktop 的
`host.docker.internal`。在 Linux 上，可为服务添加 `extra_hosts: ["host.docker.internal:host-gateway"]`，再配置模型 URL。
另一个 Compose 服务可以通过其服务名访问。

HTTP action API 没有内置认证，并包含写入和删除操作。保留默认的回环地址端口映射。需要跨主机访问时，在服务前配置
带认证的 TLS 代理，并限制服务端口的直接访问；Studio、HTTP Job 和 MCP 都应受到同样的访问限制。

镜像包含配置使用的 Agent SDK 依赖，但不会自动获取宿主机 OAuth 文件、对话记录、插件、外部 MCP 可执行程序或宿主机
工作区路径。需要的数据应显式挂载，额外插件和工具应安装在派生镜像中，使重建容器后仍能保留安装内容。
不要把凭证写入 Docker 构建参数或镜像层。可选的 FAISS/zvec 后端还依赖目标机器的能力。

## 健康检查、升级和恢复

Docker 调用已有的 `POST /health_check`，同时要求 `success=true` 和 `metadata.health.healthy=true`。
探针地址跟随实际配置和 CLI 端口覆盖，并绕过出站代理设置。初始化有 120 秒健康检查宽限期；大型工作区可在 Compose
中延长 `healthcheck.start_period`。该检查报告组件状态，不保证远端模型会接受后续请求。Docker 的 unhealthy 状态本身
不会触发 `restart: unless-stopped`；该策略只重启已经退出的进程。

升级前停止写入，并备份**完整**宿主机工作区和部署配置，再执行：

```bash
docker compose stop
# 在这里备份已配置的宿主机工作区。
docker compose pull
docker compose up -d --no-build
docker compose exec reme reme health_check
```

本地构建部署时，将 `pull` 和 `up --no-build` 替换为 `docker compose up --build -d`。Compose 为正常关闭留出 60 秒。
重建容器会保留绑定挂载的工作区；一个工作区应只运行一个 ReMe 写入进程。派生状态恢复流程见
[备份和恢复](./operations.md)，不要为修复索引删除记忆文件。

本地构建后可以验证容器：

```bash
docker build -t reme:local .
python scripts/test_docker_image.py --image reme:local
```

验证脚本使用一次性工作区，不传入模型凭证；检查 Studio、HTTP/MCP、路径隔离、非 root 执行、正常关闭，以及更换容器和
端口之后的记忆检索。
