# projects/drone-vision/lambda/Dockerfile
# ─────────────────────────────────────────────────────────
#  AWS Lambda – Python 3.11 base image
# ─────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11

# ── Application code ────────────────────────────────────
COPY lambda/ /var/task

# ── Model weights ───────────────────────────────────────
COPY model/*.onnx /var/task/model/
COPY model/*.pt   /var/task/model/
COPY model/*.yml  /var/task/model/

# ── Python deps (no wheel cache → smaller image) ────────
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# ── Lambda entry-point ──────────────────────────────────
CMD ["app.handler"]
