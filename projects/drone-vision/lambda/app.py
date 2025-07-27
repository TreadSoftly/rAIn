from io import BytesIO
from PIL import Image, ImageDraw
import base64, json, os

def handler(event, context):
    # API Gateway proxy event with base64‑encoded body
    body = base64.b64decode(event["body"])
    img  = Image.open(BytesIO(body)).convert("RGB")

    # draw a fake bounding‑box just to prove it works
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([(w*0.25, h*0.25), (w*0.75, h*0.75)], outline="red", width=4)

    buf = BytesIO(); img.save(buf, format="JPEG")
    out_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": { "Content-Type": "image/jpeg" },
        "body": out_b64,
    }
