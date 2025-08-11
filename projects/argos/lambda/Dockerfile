FROM public.ecr.aws/lambda/python:3.11

# Keep pip lean & quiet
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# ───────────────────────────────────────────────────────────────
# Install Lambda-specific Python deps FIRST (uses pinned wheels)
#   - This guarantees we pull the lambda/requirements.txt that
#     pins onnxruntime==1.16.x, numpy==1.26.x, etc.
# ───────────────────────────────────────────────────────────────
COPY lambda/requirements.txt /var/task/requirements.txt
RUN pip install --upgrade "pip<25.3" && \
    pip install --no-cache-dir --only-binary=:all: -r /var/task/requirements.txt && \
    pip check

# ───────────────────────────────────────────────────────────────
# App code
#   Copy code, then hard-prune model dir so we only ship ONNX/YAML
#   (no *.pt — keeps the image small and avoids torch in Lambda)
# ───────────────────────────────────────────────────────────────
COPY lambda/   /var/task
COPY panoptes/ /var/task/panoptes/

# Ensure we don't accidentally include *.pt from the earlier copy
RUN rm -rf /var/task/panoptes/model && mkdir -p /var/task/panoptes/model

# Model weights – ONNX only to keep image light (no Torch)
# NOTE: these COPY lines require that matching files exist locally.
COPY panoptes/model/*.onnx /var/task/panoptes/model/
# COPY panoptes/model/*.pt /var/task/panoptes/model/   # ← intentionally omitted in Lambda
COPY panoptes/model/*.yml  /var/task/panoptes/model/

# ───────────────────────────────────────────────────────────────
# Lambda entry point
# ───────────────────────────────────────────────────────────────
CMD ["app.handler"]
