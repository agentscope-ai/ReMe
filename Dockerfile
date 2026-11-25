# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS studio-builder
WORKDIR /build
COPY reme/_version.py reme/_version.py
COPY reme_studio/ reme_studio/
RUN --mount=type=cache,target=/root/.npm npm ci --prefix reme_studio --ignore-scripts
RUN npm run build:static --prefix reme_studio && test -f reme_studio/web/dist-static/index.html

FROM python:3.11-slim-bookworm AS python-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY reme/ reme/
COPY reme_studio/web/python/ reme_studio/web/python/
COPY --from=studio-builder /build/reme_studio/web/dist-static/ reme_studio/web/dist-static/
COPY scripts/package_studio.py scripts/package_studio.py
RUN python scripts/package_studio.py && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install ./reme_studio/web/python ".[core,image-heif]" \
    && python -m pip check
# Check installed resources away from the checkout, so source files cannot mask
# incomplete wheels or a Studio package fetched accidentally from PyPI.
WORKDIR /tmp
RUN python -I -c "import reme; from reme_studio import static_dir; from reme.config import resolve_app_config; assert (static_dir() / 'index.html').is_file(); assert resolve_app_config(log_config=False)['service']['backend'] == 'http'"

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libgomp1 libstdc++6 tini tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 reme \
    && useradd --uid 1000 --gid reme --no-create-home reme \
    && mkdir -p /app /data /state \
    && chown reme:reme /app /data /state \
    && chmod 700 /state
COPY --from=python-builder /opt/venv /opt/venv
COPY scripts/reme_container.py /usr/local/lib/reme_container.py
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/reme-home \
    REME_DAEMON_DIR=/state \
    REME_WORKSPACE_DIR=/data \
    REME_HOST=0.0.0.0
WORKDIR /app
USER reme
EXPOSE 2333
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "/usr/local/lib/reme_container.py", "--healthcheck"]
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/usr/local/lib/reme_container.py"]
CMD ["start"]
