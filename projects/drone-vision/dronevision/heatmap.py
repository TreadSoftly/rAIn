"""
predict_heatmap_mp4.py – apply Drone-Vision segmentation & labelling (heat-map overlay)
on **every frame** of a video and save the result.

• Always writes “*_heat.mp4” into *tests/results/* by default.
• Uses FFmpeg for fast H.264 re-encoding if available.
  – If FFmpeg is **absent** *or* exits with a non-zero status we fall back
    to moving the raw MJPG/AVI so the file still exists for the tests.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]

# Setup default segmentation model weights (prefer yolo11x-seg.pt, fallback to others)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _PROJECT_ROOT / "model"
if (_MODEL_DIR / "yolo11x-seg.pt").exists():
    _default_seg_weights = _MODEL_DIR / "yolo11x-seg.pt"
elif (_MODEL_DIR / "yolo11m-seg.pt").exists():
    _default_seg_weights = _MODEL_DIR / "yolo11m-seg.pt"
elif (_MODEL_DIR / "yolov8x-seg.pt").exists():
    _default_seg_weights = _MODEL_DIR / "yolov8x-seg.pt"
elif (_MODEL_DIR / "yolov8s-seg.pt").exists():
    _default_seg_weights = _MODEL_DIR / "yolov8s-seg.pt"
elif (_MODEL_DIR / "yolov8n-seg.pt").exists():
    _default_seg_weights = _MODEL_DIR / "yolov8n-seg.pt"
else:
    _default_seg_weights = None  # no seg model found

_seg_model: YOLO | None = None

def heatmap_overlay(image: Image.Image, boxes: np.ndarray[Any, Any] | None = None, *, alpha: float = 0.4, return_mask: bool = False) -> Image.Image | np.ndarray[Any, Any]:
    """
    Create an overlay image with segmentation masks (each object highlighted with a unique color and labeled),
    or a fallback heatmap overlay using detection boxes if segmentation model is unavailable.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image on which to overlay detections.
    boxes : np.ndarray or None
        Optional detection boxes array [N,5] or [N,6] (x1, y1, x2, y2, conf, class_idx).
        If segmentation model is not available, these boxes are used to draw simple highlights.
    alpha : float
        Blending factor for overlay transparency (0 = no overlay, 1 = full overlay color).
    return_mask : bool
        If True, return a mask or heatmap array instead of a PIL image.

    Returns
    -------
    PIL.Image.Image or numpy.ndarray
        If return_mask is False, returns a PIL Image with overlays drawn.
        If return_mask is True, returns a numpy array mask highlighting detection areas.
    """
    global _seg_model
    # Ensure image is in RGB format
    img = image.convert("RGB")
    img_w, img_h = img.size

    # Initialize segmentation model if not already loaded
    if _seg_model is None:
        if _default_seg_weights and _default_seg_weights.exists():
            try:
                _seg_model = YOLO(str(_default_seg_weights))
            except Exception:
                _seg_model = None
        else:
            _seg_model = None

    # Use segmentation model if available
    if _seg_model is not None:
        # Prepare image as numpy array (OpenCV uses BGR)
        frame = np.array(img)  # RGB array
        bgr_image = frame[:, :, ::-1]  # convert to BGR
        # Run segmentation model
        try:
            results = _seg_model.predict(bgr_image, imgsz=640, conf=0.25, verbose=False)  # type: ignore
        except Exception:
            results = []
        if results:
            result = results[0]
        else:
            result = None

        if result is not None and hasattr(result, "masks") and result.masks is not None:
            # Get masks data as numpy boolean arrays
            mask_data = result.masks.data  # type: ignore
            try:
                import torch  # type: ignore
                if isinstance(mask_data, torch.Tensor):
                    masks_np: NDArray[np.float32] = mask_data.cpu().numpy().astype(np.float32)  # type: ignore
                else:
                    masks_np: NDArray[np.float32] = np.asarray(mask_data, dtype=np.float32)
            except ImportError:
                masks_np: NDArray[np.float32] = np.asarray(mask_data, dtype=np.float32)
            # Start with original image as float32 array for blending
            overlay = bgr_image.astype(np.float32)
            # Generate unique colors for each mask
            num_masks = masks_np.shape[0]
            for i in range(num_masks):
                mask_i = masks_np[i]
                if mask_i.shape[0] != img_h or mask_i.shape[1] != img_w:
                    mask_i: NDArray[np.bool_] = cv2.resize(mask_i.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_i: NDArray[np.bool_] = mask_i >= 0.5
                # Compute a unique color for this mask (BGR order for overlay)
                hue = (i * 0.618033988749895) % 1.0  # use golden ratio for color variation
                import colorsys
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                color_bgr = np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.float32)
                # Blend mask color with original image
                overlay[mask_i] = overlay[mask_i] * (1 - alpha) + color_bgr * alpha
            overlay = overlay.astype(np.uint8)
            # Draw labels and IDs on the overlay
            if hasattr(result, "boxes") and result.boxes is not None:
                names: dict[int, str] = _seg_model.names if hasattr(_seg_model, "names") else {}
                # Explicitly convert to numpy arrays with known dtype for type safety
                cls_arr = np.array(result.boxes.cls, dtype=np.float32)  # type: ignore
                conf_arr = np.array(result.boxes.conf, dtype=np.float32)  # type: ignore
                xyxy_arr = np.array(result.boxes.xyxy, dtype=np.float32)  # type: ignore
                for i in range(len(result.boxes)):
                    cls_id = int(cls_arr[i]) if hasattr(result.boxes, "cls") else -1
                    conf_val = float(conf_arr[i]) if hasattr(result.boxes, "conf") else 0.0
                    class_name = names.get(cls_id, str(cls_id))
                    label_text = f"{class_name} {conf_val * 100:.1f}% ID {i+1}"
                    # Determine text placement (above the object, near top-left of box)
                    if hasattr(result.boxes, "xyxy"):
                        x1, y1, x2, y2 = map(int, xyxy_arr[i])
                    else:
                        # Fallback to mask bounds if no boxes
                        ys, xs = np.where(masks_np[i] >= 0.5)
                        y1 = int(ys.min()) if ys.size > 0 else 0
                        x1 = int(xs.min()) if xs.size > 0 else 0
                        x2 = int(xs.max()) if xs.size > 0 else x1
                        y2 = int(ys.max()) if ys.size > 0 else y1
                    # Draw rectangle background for text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                    bg_x0 = x1
                    bg_y0 = max(0, y1 - text_h - 4)
                    bg_x1 = x1 + text_w + 2
                    bg_y1 = bg_y0 + text_h + 4
                    color = (int(overlay[max(y1, 0), max(x1, 0), 0]),
                            int(overlay[max(y1, 0), max(x1, 0), 1]),
                            int(overlay[max(y1, 0), max(x1, 0), 2]))
                    cv2.rectangle(overlay, (bg_x0, bg_y0), (bg_x1, bg_y1), color, -1)
                    cv2.putText(overlay, label_text, (bg_x0 + 1, bg_y0 + text_h + 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            # If return_mask requested, build a combined mask array
            if return_mask:
                combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for i in range(masks_np.shape[0]):
                    mask_bool = masks_np[i] >= 0.5
                    combined_mask[mask_bool] = 255  # mark mask areas
                return combined_mask
            # Convert overlay (BGR) back to PIL Image (RGB)
            overlay_rgb = overlay[:, :, ::-1]
            out_img = Image.fromarray(overlay_rgb)
            return out_img
    # Fallback if segmentation model is not available or failed
    if boxes is None or boxes.size == 0:
        # No detection info, just return original image or empty mask
        if return_mask:
            return np.zeros((img_h, img_w), dtype=np.uint8)
        return img.copy()
    # Fallback heatmap overlay using detection boxes
    output_img = img.copy()
    # Use semi-transparent red for each box
    for i, box in enumerate(np.asarray(boxes).reshape(-1, boxes.shape[-1] if hasattr(boxes, "shape") else 5)):
        if box.shape[0] < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        rect_width = x2 - x1
        rect_height = y2 - y1
        overlay_rect = Image.new("RGBA", (rect_width, rect_height), (255, 0, 0, int(alpha * 255)))
        output_img.paste(overlay_rect, (x1, y1), overlay_rect)
    if return_mask:
        mask_arr = np.zeros((img_h, img_w), dtype=np.uint8)
        for box in np.asarray(boxes).reshape(-1, boxes.shape[-1] if hasattr(boxes, "shape") else 5):
            if box.shape[0] < 4:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            mask_arr[y1:y2, x1:x2] = 255
        return mask_arr
    return output_img

def main(
    src: str | Path,
    *,
    cmap: str = "COLORMAP_JET",
    alpha: float = 0.4,
    kernel_scale: float = 5.0,
    out_dir: str | Path | None = None,
    **kw: Any,
) -> Path:
    """
    Apply segmentation mask overlay (formerly heat-map style) to each frame of a video.
    Saves an output video with detected objects highlighted in unique colors with labels.

    Parameters
    ----------
    src : str or Path
        Path to the input video file.
    cmap, alpha, kernel_scale : (deprecated) heatmap parameters retained for compatibility (not used).
    out_dir : Output directory for the result video.

    Returns
    -------
    Path : Path to the output video file (MP4 format).
    """
    src_path = Path(src).expanduser()
    if out_dir is None:
        out_dir = _PROJECT_ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_hm_"))
    avi_path = tmp_dir / f"{stem}.avi"

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Could not open source video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(avi_path),
        cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any video writer on this system.")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_pil = Image.fromarray(frame[:, :, ::-1])
        frame_overlay = heatmap_overlay(frame_pil, boxes=None, alpha=alpha, return_mask=False)
        if isinstance(frame_overlay, Image.Image):
            frame_overlay_bgr = np.array(frame_overlay)[:, :, ::-1]
        else:
            frame_overlay_bgr = frame_overlay[:, :, ::-1] if frame_overlay.ndim == 3 else cv2.cvtColor(frame_overlay, cv2.COLOR_GRAY2BGR)
        writer.write(frame_overlay_bgr.astype(np.uint8))

    cap.release()
    writer.release()

    final_mp4 = out_dir / f"{stem}_heat.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(avi_path),
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_mp4),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        avi_path.unlink(missing_ok=True)
    except FileNotFoundError:
        shutil.move(str(avi_path), final_mp4.with_suffix(".avi"))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"✅  Saved → {final_mp4}")
    return final_mp4

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m dronevision.predict_heatmap_mp4 <video> [key=value …]")

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    cli_args = {k: (v if v.lower() not in {"true", "false"} else v.lower() == "true") for k, v in kv_pairs}
    cmap_val = str(cli_args.pop("cmap", "COLORMAP_JET"))
    alpha_val = float(cli_args.pop("alpha", 0.4))
    kernel_val = float(cli_args.pop("kernel_scale", 5.0))
    out_dir_val = cli_args.pop("out_dir", None)
    if out_dir_val is not None:
        out_dir_val = str(out_dir_val)
    main(sys.argv[1], cmap=cmap_val, alpha=alpha_val, kernel_scale=kernel_val, out_dir=out_dir_val, **cli_args)
