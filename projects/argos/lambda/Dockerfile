# ──────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11

## Application code (context = projects/argos)
COPY lambda/               /var/task
COPY panoptes/             /var/task/panoptes/

## Model weights – ONNX only to keep image light (no Torch)
COPY panoptes/model/*.onnx /var/task/panoptes/model/
# COPY panoptes/model/*.pt /var/task/panoptes/model/   # ← leave out in Lambda
COPY panoptes/model/*.yml  /var/task/panoptes/model/

# pip hygiene (avoid cache bloat; quiet version checks)
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ── minimal build tools for packages that ship C++ sources ───
# (Wheels should satisfy most deps; keep/remove as your code needs.)
RUN yum install -y gcc-c++ make && yum clean all || true

# ── python deps ───────────────────────────────────────────────
RUN pip install --upgrade "pip<25.3" && \
    pip install --no-cache-dir -r /var/task/requirements.txt && \
    pip check

# (optional) strip build tools again – keeps image small
RUN yum remove -y gcc-c++ make && yum clean all || true

# ── Lambda entry point ───────────────────────────────────────
CMD ["app.handler"]
