import base64
import io
import json
import os
import urllib.request
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as rt

# ---- model -----------------------------------------------------------------
MODEL = os.path.join(os.getcwd(), "model", "yolov8n.onnx")
session = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
in_name = session.get_inputs()[0].name
STRIDE  = 640

# ---- helpers ---------------------------------------------------------------
def fetch(url: str, timeout=10) -> bytes:
    """Download bytes from URL with a desktop UA (Unsplash blocks curl)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def detect(img: Image.Image) -> Image.Image:
    img_rs = img.resize((STRIDE, STRIDE))
    x = np.asarray(img_rs).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    pred = np.array(session.run(None, {in_name: x})[0])[0]   # (84, 8400)

    scores = pred[4]
    best   = int(scores.argmax())
    if scores[best] > 0.40:
        x1, y1, x2, y2 = pred[:4, best]

        # --- sanitise -------------------------------------------------------
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:   # 1 px or less → ignore
            return img
        # -------------------------------------------------------------------

        scale = np.array([img.width, img.height] * 2) / STRIDE
        box   = (np.array([x1, y1, x2, y2]) * scale).tolist()
        ImageDraw.Draw(img).rectangle(box, outline="red", width=3)

    return img

# ---- Lambda handler --------------------------------------------------------
def handler(event, _context):
    try:
        body_str = event.get("body", "") or ""
        if event.get("isBase64Encoded"):              # S3 trigger
            img_bytes = base64.b64decode(body_str)
        else:
            try:                                      # try JSON first
                data = json.loads(body_str)
                img_bytes = fetch(data["image_url"])
            except (json.JSONDecodeError, KeyError):
                img_bytes = base64.b64decode(body_str)

        img_out = detect(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        buf = io.BytesIO()
        img_out.save(buf, format="JPEG")
        return {
            "statusCode": 200,
            "isBase64Encoded": True,
            "headers": {"Content-Type": "image/jpeg"},
            "body": base64.b64encode(buf.getvalue()).decode(),
        }

    except Exception as exc:
        # Anything goes wrong → plain text 500 so you see the error
        return {"statusCode": 500, "body": str(exc)}
