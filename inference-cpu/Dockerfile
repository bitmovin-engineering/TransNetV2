# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS uv
FROM python:3.12-slim@sha256:6d2b0577e6d1632f077331de8e4780535f54839d163ded7abb4b7dd37b5d37ac
COPY inference-cpu/pyproject.toml inference-cpu/uv.lock inference-cpu/LICENSE /opt/transnet-package/
COPY inference-cpu/src/ /opt/transnet-package/src/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    cd /opt/transnet-package \
    && UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/transnet-venv \
       SOURCE_DATE_EPOCH=315532800 uv sync --frozen --no-dev --no-editable --python python3 \
    && rm -rf /opt/transnet-package
ENTRYPOINT ["/opt/transnet-venv/bin/python", "-m", "transnetv2_cpu.worker"]
