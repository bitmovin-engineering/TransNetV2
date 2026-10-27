# syntax=docker/dockerfile:1
FROM python:3.12-slim
COPY inference-cpu/pyproject.toml inference-cpu/LICENSE /opt/transnet-package/
COPY inference-cpu/src/ /opt/transnet-package/src/
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --no-cache-dir /opt/transnet-package \
    && rm -rf /opt/transnet-package
ENTRYPOINT ["python", "-m", "transnetv2_cpu.worker"]
