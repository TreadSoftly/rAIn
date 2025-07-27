# projects/drone-vision/lambda/Dockerfile
# ------------------------------------------------------------
#  Python 3.10 Lambda base image
# ------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.10

# ------------------------------------------------------------
#  Copy application code
#    Everything in lambda/ → /var/task   (Lambda workdir)
# ------------------------------------------------------------
COPY . /var/task

# ------------------------------------------------------------
#  Copy ALL ONNX weights into /var/task/model/
# ------------------------------------------------------------
COPY model/*.onnx /var/task/model/

# ------------------------------------------------------------
#  Install Python dependencies (no wheel cache to keep image small)
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# ------------------------------------------------------------
#  Lambda entry‑point
# ------------------------------------------------------------
CMD ["app.handler"]
