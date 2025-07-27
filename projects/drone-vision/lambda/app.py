import base64
import io
import json
import os
import urllib.request
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as rt

# --- load model once --------------------------------------------------------
MODEL_PATH = os.path.join(os.getcwd(), "model", "yolov8n.onnx")
session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
in_name = session.get_inputs()[0].name
STRIDE  = 640  # model input size
# ---------------------------------------------------------------------------

def _img_from_event(event):
    """Accept either JSON {"image_url": "..."}  or raw base‑64 body."""
    try:
        body = event["body"]
        # If API Gateway used base64‑encoding (proxy integrations) it flags it:
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body)
            return Image.open(io.BytesIO(body)).convert("RGB")

        # If body is JSON with an URL
        maybe_json = json.loads(body)
        if isinstance(maybe_json, dict) and "image_url" in maybe_json:
            with urllib.request.urlopen(maybe_json["image_url"]) as resp:
                return Image.open(resp).convert("RGB")
    except Exception:
        pass  # fall through ↴

    # Otherwise assume plain base‑64‑text
    return Image.open(io.BytesIO(base64.b64decode(body))).convert("RGB")

def handler(event, _context):
    try:
        img = _img_from_event(event)

        # --- minimal preprocessing / inference -----------------------------
        img_rs = img.resize((STRIDE, STRIDE))
        x = np.asarray(img_rs).transpose(2,0,1)[None].astype(np.float32) / 255.0
        pred_raw = session.run(None, {in_name: x})[0]
        pred = np.array(pred_raw)[0]           # (84,8400)

        # simple “take best” post‑process
        scores = pred[4]
        best   = np.argmax(scores)
        if scores[best] > 0.4:
            x1,y1,x2,y2 = pred[:4, best]
            scale       = np.array([img.width, img.height]*2) / STRIDE
            box         = (np.array([x1,y1,x2,y2])*scale).tolist()
            ImageDraw.Draw(img).rectangle(box, outline="red", width=3)
        # -------------------------------------------------------------------

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return {
            "statusCode": 200,
            "isBase64Encoded": True,
            "headers": {"Content-Type": "image/jpeg"},
            "body": base64.b64encode(buf.getvalue()).decode(),
        }

    except Exception as e:          # any failure ➜ log and 500
        print("ERROR:", e)
        return {"statusCode": 500, "body": str(e)}
