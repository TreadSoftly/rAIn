
# projects/argos/lambda/Dockerfile
# ──────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11

# ── application code (context = projects/argos) ───────
COPY lambda/               /var/task

# ── model weights  (inside the package now)  ─────────────────
COPY panoptes/model/*.onnx /var/task/panoptes/model/
COPY panoptes/model/*.pt   /var/task/panoptes/model/
COPY panoptes/model/*.yml  /var/task/panoptes/model/

# ── minimal build tools for packages that ship C++ sources ───
RUN yum install -y gcc-c++ make && yum clean all

# ── python deps ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# (optional) strip build tools again – keeps image small
RUN yum remove -y gcc-c++ make && yum clean all

# ── Lambda entry point ───────────────────────────────────────
CMD ["app.handler"]
