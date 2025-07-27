import base64
import io
import os
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as rt

# Load model once at cold‑start
MODEL_PATH = os.path.join(os.getcwd(), "model", "yolov8n.onnx")
session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
in_name  = session.get_inputs()[0].name
stride   = 640  # model input

def handler(event, context):
    """API Gateway / S3 event → annotated JPEG"""
    b64 = event["body"] if isinstance(event["body"], str) else ""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    # Pre‑process
    img_rs = img.resize((stride, stride))
    x = np.asarray(img_rs).transpose(2,0,1)[None].astype(np.float32) / 255.0

    # Inference
    pred = session.run(None, {in_name: x})[0]            # (1,84,8400)
    pred = np.array(pred)[0]                             # (84,8400)

    # Very rough post‑processing: find highest‑score object
    scores = pred[4]
    best   = np.argmax(scores)
    if scores[best] > 0.4:                               # threshold
        x1,y1,x2,y2 = pred[:4, best]
        scale = np.array([img.width, img.height]*2) / stride
        box   = (np.array([x1,y1,x2,y2]) * scale).tolist()
        draw  = ImageDraw.Draw(img)
        draw.rectangle(box, outline="red", width=3)

    # Return JPEG bytes
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": {"Content-Type": "image/jpeg"},
        "body": base64.b64encode(buf.getvalue()).decode()
    }
