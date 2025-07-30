# projects/drone-vision/lambda/Dockerfile
# ------------------------------------------------------------------
#  AWS Lambda - Python 3.11 base image
# ------------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.11

# ------------------------------------------------------------------
#  Copy application code
#     Everything in lambda/  →  /var/task   (Lambda work-dir)
# ------------------------------------------------------------------
COPY . /var/task

# ------------------------------------------------------------------
#  Copy ONNX and (optionally) PyTorch weights
#     The files live **one directory up** in model/
# ------------------------------------------------------------------
COPY ../model/*.onnx  /var/task/model/
COPY ../model/*.pt    /var/task/model/
COPY ../model/*.yml  /var/task/model/

# ------------------------------------------------------------------
#  Install runtime dependencies
#  (disable wheel cache - keeps the image small)
# ------------------------------------------------------------------
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# ------------------------------------------------------------------
#  Lambda entry-point
# ------------------------------------------------------------------
CMD ["app.handler"]
