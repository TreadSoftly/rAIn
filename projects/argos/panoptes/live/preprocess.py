"""
Reusable frame preprocessing helpers for the live pipeline.

Performs color conversion, letterbox resize, and normalization in a single pass
and keeps scratch buffers around so we avoid per-frame allocations.  The API
is intentionally lightweight: provide the desired output size and device, then
call ``process(images, predictor)`` to obtain a torch tensor ready for the
Ultralytics predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, SupportsInt, cast

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from numpy.typing import NDArray  # type: ignore
except Exception:  # pragma: no cover
    NDArray = Any  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from torch import Tensor  # type: ignore
else:
    Tensor = Any  # type: ignore[assignment]

try:
    from ultralytics.utils.ops import LetterBox  # type: ignore
except Exception:  # pragma: no cover
    LetterBox = None  # type: ignore


if TYPE_CHECKING:
    from numpy.typing import NDArray as _NDArray

    UInt8Array = _NDArray[np.uint8]
    FloatArray = _NDArray[np.float32]
else:  # pragma: no cover - runtime fallback
    UInt8Array = np.ndarray  # type: ignore[misc, assignment]
    FloatArray = np.ndarray  # type: ignore[misc, assignment]


@dataclass
class _PreprocessConfig:
    target_hw: Tuple[int, int]  # (height, width)
    device: str = "cpu"
    normalize: bool = True
    rgb: bool = True
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    stride: int = 32


class Preprocessor:
    """
    Consolidated resize/letterbox/color conversion utility with buffer reuse.

    Parameters
    ----------
    target_size:
        Desired output size as (width, height).  If ``None`` we use (640, 640).
    stride:
        Model stride used for letterboxing (defaults to 32).
    device:
        ``"cpu"``, ``"gpu"``, or ``"auto"``.  GPU path uses cv2.cuda if available.
    normalize:
        Divide pixel values by 255 when True.
    rgb:
        Convert BGR→RGB before returning.
    mean/std:
        Optional per-channel normalization (values in 0-1 space).
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        *,
        stride: int = 32,
        device: str = "cpu",
        normalize: bool = True,
        rgb: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for live preprocessing.")
        if torch is None:
            raise RuntimeError("PyTorch is required for live preprocessing.")
        if LetterBox is None:
            raise RuntimeError("Ultralytics utilities are required for preprocessing.")

        if target_size is None:
            w, h = 640, 640
        else:
            if len(target_size) != 2:
                raise ValueError("target_size must contain exactly two values.")
            w, h = int(target_size[0]), int(target_size[1])

        self.cfg = _PreprocessConfig(
            target_hw=(int(h), int(w)),
            device=device.lower(),
            normalize=normalize,
            rgb=rgb,
            mean=mean,
            std=std,
            stride=int(stride) if stride else 32,
        )
        self._scale = 1.0 / 255.0 if normalize else 1.0
        self._mean = (
            np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
            if mean is not None
            else None
        )
        self._std = (
            np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
            if std is not None
            else None
        )
        self._letterbox: Any = LetterBox(  # type: ignore[call-arg]
            self.cfg.target_hw,
            auto=False,
            stride=self.cfg.stride,
        )
        self._batch_buffer: Optional[FloatArray] = None
        self._batch_capacity: int = 0

    # ------------------------------------------------------------------ helpers
    def _ensure_buffer(self, batch: int) -> FloatArray:
        th, tw = self.cfg.target_hw
        if self._batch_buffer is None or self._batch_capacity < batch:
            self._batch_buffer = np.empty((batch, 3, th, tw), dtype=np.float32)
            self._batch_capacity = batch
        buffer = self._batch_buffer
        assert buffer is not None
        return buffer

    def _convert_single(self, dest: FloatArray, image: UInt8Array) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV not available for preprocessing.")
        image_u8 = np.asarray(image, dtype=np.uint8)
        if image_u8.ndim == 2:
            image_u8 = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
        elif image_u8.ndim == 3 and image_u8.shape[2] == 1:
            image_u8 = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
        processed = np.asarray(self._letterbox(image=image_u8))
        if self.cfg.rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        view: FloatArray = processed.transpose(2, 0, 1).astype(np.float32, copy=False)
        dest[...] = view
        if self._scale != 1.0:
            dest *= self._scale
        if self._mean is not None:
            dest -= self._mean
        if self._std is not None:
            dest /= self._std  # type: ignore[assignment]

    def _to_device(self, tensor: Tensor, predictor: Any) -> Tensor:
        device = getattr(predictor, "device", None)
        if self.cfg.device == "auto":
            use_gpu = device is not None and "cuda" in str(device).lower()
        else:
            use_gpu = self.cfg.device == "gpu"
        if use_gpu and device is not None:
            return tensor.to(device, non_blocking=True)
        return tensor.to(device or "cpu")

    # ------------------------------------------------------------------ public
    def process(
        self, images: Sequence[Optional[UInt8Array]], predictor: Any
    ) -> Tensor:
        """
        Run preprocessing for a batch of images.

        Parameters
        ----------
        images:
            Sequence of BGR frames (H, W, 3) straight from capture.
        predictor:
            Ultralytics predictor instance (provides device + fp16 hints).
        """
        if not images:
            raise ValueError("Preprocessor requires at least one image.")

        buffer = self._ensure_buffer(len(images))
        for idx, image in enumerate(images):
            if image is None:
                raise ValueError("Encountered None image in batch.")
            self._convert_single(buffer[idx], image)

        if torch is None:
            raise RuntimeError("PyTorch not available for preprocessing.")
        tensor = torch.from_numpy(buffer[: len(images)])  # type: ignore[arg-type]
        fp16 = bool(getattr(getattr(predictor, "model", None), "fp16", False))
        tensor = tensor.half() if fp16 else tensor.float()
        tensor = self._to_device(tensor, predictor)
        return tensor


def attach_preprocessor(
    wrapper: Any,
    *,
    target_size: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> Optional[Preprocessor]:
    """
    Attach a reusable preprocessor to the underlying Ultralytics predictor.
    Returns the created ``Preprocessor`` (or ``None`` if the model/predictor
    cannot be resolved).
    """
    model = wrapper.active_model()
    if model is None:
        return None

    predictor = getattr(model, "predictor", None)
    if predictor is None:
        try:
            predictor = model.predictor  # type: ignore[attr-defined]
        except Exception:
            predictor = None
    if predictor is None:
        return None

    raw_stride = getattr(model, "stride", 32)
    if isinstance(raw_stride, np.ndarray):
        flat_stride = raw_stride.astype(int, copy=False).reshape(-1)
        stride_val = int(flat_stride[0]) if flat_stride.size else 32
    elif isinstance(raw_stride, (list, tuple)):
        stride_val = int(cast(SupportsInt, raw_stride[0])) if raw_stride else 32
    else:
        stride_val = int(cast(SupportsInt, raw_stride))

    resolved_device = (device or "cpu").strip().lower()
    if resolved_device not in {"cpu", "gpu", "auto"}:
        resolved_device = "cpu"

    resolved_size = target_size
    imgsz = getattr(getattr(predictor, "args", None), "imgsz", None)
    if resolved_size is None and imgsz is not None:
        if isinstance(imgsz, np.ndarray):
            dims = [int(x) for x in imgsz.astype(int, copy=False).reshape(-1).tolist()]
            if len(dims) == 1:
                resolved_size = (dims[0], dims[0])
            elif len(dims) >= 2:
                resolved_size = (dims[1], dims[0])
        elif isinstance(imgsz, (list, tuple)):
            dims: List[int] = []
            typed_imgsz = cast(Sequence[SupportsInt | None], imgsz)
            for raw_value in typed_imgsz:
                if raw_value is None:
                    continue
                dims.append(int(raw_value))
            if len(dims) == 1:
                resolved_size = (dims[0], dims[0])
            elif len(dims) >= 2:
                resolved_size = (dims[1], dims[0])
        elif isinstance(imgsz, (int, float)):
            dim = int(imgsz)
            resolved_size = (dim, dim)

    preproc = Preprocessor(
        target_size=resolved_size,
        stride=stride_val,
        device=resolved_device,
        mean=mean,
        std=std,
    )

    def _patched_preprocess(im: List[UInt8Array]):
        return preproc.process(im, predictor)

    predictor._argos_preprocessor = preproc  # type: ignore[attr-defined]
    predictor.preprocess = _patched_preprocess  # type: ignore[assignment]

    # Keep Ultralytics internals aware of the desired image size.
    th, tw = preproc.cfg.target_hw
    imgsz = [th, tw]
    if hasattr(predictor, "args") and getattr(predictor.args, "imgsz", None):
        predictor.args.imgsz = imgsz  # type: ignore[attr-defined]
    elif hasattr(predictor, "imgsz"):
        predictor.imgsz = imgsz  # type: ignore[attr-defined]

    return preproc
