---
title: Docker Deployment
description: Run ReMe and Studio in Docker with a user-owned persistent workspace.
---

# Docker Deployment

The image includes ReMe's `core` and HEIF image dependencies and the Studio static frontend. One HTTP process serves the API,
Studio at `/`, and MCP at `/mcp`. The default configuration keeps embeddings disabled; file operations and BM25 search do
not require model credentials.

## Build and run with Compose

Use Docker Engine or Docker Desktop with Compose **2.24.0 or newer**. From the repository root:

```bash
mkdir -p .reme
docker compose up --build -d
docker compose logs -f reme
```

Open <http://127.0.0.1:2333>. Compose binds the host port to loopback and mounts `./.reme` at `/data`. The source checkout and
Studio assets are not mounted over the installed application.

On Linux, if your workspace is not owned by UID/GID 1000, set the process identity before starting:

```bash
export REME_UID=$(id -u)
export REME_GID=$(id -g)
docker compose up --build -d
```

For model-powered memory evolution, copy `deploy/docker/example.env` to `.env` if you do not already have one, then fill in
your model credentials. Compose injects this optional file at runtime; Docker builds exclude `.env` files. Compose's
`--env-file` controls variable interpolation; `REME_ENV_FILE` selects the file injected into the container.

## Use a published image

The Docker workflow publishes `ghcr.io/agentscope-ai/reme:main` after successful main-branch checks. Stable GitHub releases
publish their package version and `latest`; prereleases do not update `latest`. Both Linux amd64 and arm64 images are tested
before their combined tags are published. Publication starts when the workflow is enabled in the repository.

```bash
mkdir -p "$HOME/.reme"
docker run -d --name reme \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:2333:2333 \
  --mount "type=bind,source=$HOME/.reme,target=/data" \
  --restart unless-stopped \
  ghcr.io/agentscope-ai/reme:main
```

Add `--env-file /path/to/model.env` before the image name when using model credentials. For reproducible deployments,
replace `main` with a released version or image digest. To use the image with Compose, set `REME_IMAGE`, then run
`docker compose pull` and `docker compose up -d --no-build`.

## Paths, configuration, and ports

The image runs as UID/GID 1000 by default. `/data` contains the entire workspace: source sessions, resources, daily notes,
digest notes, and rebuildable metadata. Create the host directory yourself and make it writable by the configured user.
Mounting only `metadata/` does not preserve the source memories. Paths in a custom configuration refer to the container's
filesystem; additional paths need additional mounts.

| Setting | Meaning |
|---|---|
| `REME_WORKSPACE_DIR` | Container workspace; image default `/data`, fixed to `/data` by Compose |
| `REME_CONFIG` | Existing config name or mounted YAML/JSON path; unset uses the built-in default |
| `REME_HOST` | HTTP bind address; image and Compose use `0.0.0.0` |
| `REME_PORT` | Container port override; Compose defaults to `2333` |
| `REME_TIMEZONE` | Optional application timezone override; otherwise the application default applies |
| `REME_DATA_DIR` | Compose host workspace directory; default `./.reme` |
| `REME_PUBLISHED_PORT` | Compose host port; default `2333`, independent of the container port |
| `REME_BIND_ADDRESS` | Compose host bind address; default `127.0.0.1` |
| `REME_UID`, `REME_GID` | Compose process identity; both default to `1000` |
| `REME_ENV_FILE` | Optional Compose runtime environment file; default `.env` |

Explicit `start key=value` arguments override container environment settings, which override the loaded configuration for
those keys. Other keys retain ReMe's normal deep merge behavior. File logging defaults to off in the image; use container
logs. `log_to_file=true` explicitly enables file logs under `/app/logs`, which requires a separate mount to persist them.
The temporary home directory `/tmp/reme-home` and probe address file are disposable, not workspace storage.

To customize the full job/component configuration, copy `reme/config/default.yaml` to `reme.yaml`, edit it, set
`REME_CONFIG=/etc/reme/config.yaml` in `.env`, and add `compose.override.yaml`:

```yaml
services:
  reme:
    volumes:
      - ./reme.yaml:/etc/reme/config.yaml:ro
```

Keep the `health_check` Job enabled and in `service.jobs` if you use an allowlist. The image's probe uses HTTP; when
overriding the service to CLI or MCP stdio, disable the Docker health check with `--no-healthcheck` (or Compose
`healthcheck: {disable: true}`).

To override startup without changing the image:

```bash
docker run --rm -p 127.0.0.1:2444:2444 \
  --mount "type=bind,source=$HOME/.reme,target=/data" \
  reme:local start service.port=2444 timezone=UTC
docker compose exec reme reme health_check
docker compose exec reme reme status
```

Other commands pass through unchanged, including `reme start job=version` for a one-shot job or `python` for diagnostics.
From a host CLI, supply the published address explicitly, for example `reme health_check host=127.0.0.1 port=2444`.
Host process discovery cannot reconstruct a container's startup arguments. Configure agent integrations to use the
published HTTP or MCP endpoint instead of starting a second native ReMe on the same workspace.

## Networking and optional tools

`127.0.0.1` inside a container refers to that container. A model server on the host needs a reachable host address, such as
`host.docker.internal` on Docker Desktop. On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` to the service
and configure the model URL accordingly. Another Compose service is reachable by its service name.

The HTTP action API has no built-in authentication and includes write/delete operations. Keep the default loopback port
publication. For access from another machine, place an authenticated TLS proxy in front of the service and restrict direct
access to its port; the same restriction must cover Studio, HTTP Jobs, and MCP.

The image includes the configured agent SDK dependencies, but host OAuth files, transcripts, plugins, external MCP
executables, and host workspace paths are not automatically available. Mount required data explicitly and install extra
plugins/tools in a derived image so that recreating the container preserves the installation. Keep credentials out of
Docker build arguments and layers. Optional FAISS/zvec backends also depend on the capabilities of the target machine.

## Health, upgrade, and recovery

Docker posts to the existing `/health_check` Job and requires both `success=true` and `metadata.health.healthy=true`.
The probe address follows effective configuration and CLI port overrides, and bypasses outbound proxy settings.
Initialization has a 120-second health grace period; larger workspaces may need a longer Compose `healthcheck.start_period`.
This reports component health, not whether a remote model will accept a future request. An unhealthy Docker status alone
does not trigger `restart: unless-stopped`; that policy restarts exited processes.

Before upgrading, stop writes and back up the **complete** host workspace and your deployment configuration. Then:

```bash
docker compose stop
# Back up the configured host workspace here.
docker compose pull
docker compose up -d --no-build
docker compose exec reme reme health_check
```

For a locally built deployment, replace `pull` and `up --no-build` with `docker compose up --build -d`. Compose allows
60 seconds for orderly shutdown. Recreating containers leaves the bind-mounted workspace intact; use one ReMe writer
process per workspace. See [backup and recovery](./operations.md) for restoring derived state without deleting memory.

For container validation after a local build:

```bash
docker build -t reme:local .
python scripts/test_docker_image.py --image reme:local
```

The smoke check uses a disposable workspace and no model credentials. It verifies Studio, HTTP and MCP, file containment,
non-root execution, graceful shutdown, and memory search after replacing a container on a different port.
