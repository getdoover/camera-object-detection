# syntax=docker/dockerfile:1
#
# Object detection app -- runs ONNX models on the Doovit CPU.
#
# Why this does NOT use spaneng/doover_device_base
# ------------------------------------------------
# That base is now Alpine/musl. Its Python lives at /usr/local (3.11) while
# Alpine's apk `py3-*` packages (py3-onnxruntime, py3-opencv) install for
# Alpine's *system* Python 3.12 under /usr/lib/python3.12 -- a different ABI, so
# the app's interpreter can't import them. PyPI publishes no musllinux wheels for
# onnxruntime or opencv, so there is no way to get an inference stack onto that
# base short of compiling onnxruntime for musl/3.11.
#
# Debian bookworm has manylinux aarch64 wheels for everything here, so this app
# is built on python:3.11-slim-bookworm and re-declares the labels/healthcheck the
# app controller looks for. Everything else (pydoover, DDA over gRPC) is
# base-image independent.
FROM python:3.11-slim-bookworm AS base
LABEL com.doover.app="true"
LABEL com.doover.managed="true"
HEALTHCHECK --interval=30s --timeout=2s --start-period=5s CMD curl -f "127.0.0.1:$HEALTHCHECK_PORT" || exit 1
WORKDIR /app

# curl: the healthcheck above. libglib2.0-0: opencv-python-headless links it even
# in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# --- builder: resolve deps into /app/.venv -----------------------------------
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

RUN uv venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# --- runtime -----------------------------------------------------------------
FROM base AS runtime
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Where yolo.py looks for the detection weights (committed under models/).
ENV OBJECT_DETECTION_MODEL_DIR=/app/models

# No OCR pre-cache step, deliberately: the plate-OCR weights ship in models/ alongside
# the detectors and load by explicit path (see common/detectors/anpr.py). This used to
# warm fast-plate-ocr's hub cache instead, which resolves from `Path.home()` with no env
# override -- fragile here (any non-root USER breaks it) and outright broken in the
# cloud processor, where Lambda sets HOME=/tmp and every plate came back unread.

# Keep the math libraries single-threaded. The models run one image at a time on a
# 4-core CM4 shared with every other app on the device; letting BLAS/onnxruntime
# fan out to all cores just starves the camera apps mid-snapshot for no
# throughput gain at this workload.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["doover-app-run"]
