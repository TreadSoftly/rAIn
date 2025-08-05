# projects/drone-vision/lambda/Dockerfile
# ──────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11

# ── application code (context = projects/drone-vision) ───────
COPY lambda/ /var/task

# ── model weights ─────────────────────────────────────────────
COPY model/*.onnx /var/task/model/
COPY model/*.pt   /var/task/model/
COPY model/*.yml  /var/task/model/

# ── python deps ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# ── Lambda entry point ───────────────────────────────────────
CMD ["app.handler"]
