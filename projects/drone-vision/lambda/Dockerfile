# projects/drone-vision/lambda/Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# --- Python deps for inference -------------------------
RUN pip install --no-cache-dir \
    onnxruntime==1.16.3 \
    numpy==1.26.4 \
    pillow==11.3.0

# --- App code & model ----------------------------------
COPY app.py              .
COPY model/yolov8n.onnx  ./model/

CMD ["app.handler"]

# --- Python deps -------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
