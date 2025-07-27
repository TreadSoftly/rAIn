FROM public.ecr.aws/lambda/python:3.10

# Python deps for inference
RUN pip install --no-cache-dir \
        onnxruntime==1.17.3 \
        numpy==1.26.4 \
        pillow==11.3.0

# Copy code and model
COPY app.py              .
COPY model/yolov8n.onnx  ./model/

CMD ["app.handler"]
