"""
Helpers to fingerprint model artifacts (ONNX, TensorRT engines, Torch weights)
so downstream selection logic can decide which post-processing path to use.

The analyser is intentionally permissive: it never raises and only records the
best-effort information that could be derived from an artifact.  Callers should
expect missing fields and fall back to reasonable defaults.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ARTIFACT_METADATA_VERSION = 1

_NMS_OP_NAMES = {
    "nonmaxsuppression",
    "batchednms_trt",
    "efficientnms_trt",
    "multiclassnms_tensorrt",
    "multiclassnms_trt",
    "multiclassnms_dynamic_tensorrt",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _flatten_to_list(array: Any) -> list[Any]:
    try:
        array = array.reshape(-1)
    except Exception:
        pass
    try:
        return list(array.tolist())  # type: ignore[arg-type]
    except Exception:
        try:
            return list(array)
        except Exception:
            return []


def _resolve_scalar_from_initializer(initializers: Dict[str, Any], name: str) -> Optional[float]:
    if not name:
        return None
    tensor = initializers.get(name)
    if tensor is None:
        return None
    try:
        from onnx import numpy_helper  # type: ignore

        raw_array = numpy_helper.to_array(tensor)  # type: ignore[no-any-return]
        values = _flatten_to_list(raw_array)
    except Exception:
        return None
    if len(values) != 1:
        return None
    try:
        return float(values[0])
    except Exception:
        return None


def _extract_constant_scalars(graph: Any) -> Dict[str, Any]:
    scalars: Dict[str, Any] = {}
    try:
        from onnx import numpy_helper  # type: ignore
    except Exception:
        return scalars
    for node in getattr(graph, "node", []):
        if node.op_type != "Constant":
            continue
        if not node.output:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            try:
                raw_array = numpy_helper.to_array(attr.t)  # type: ignore[no-any-return]
                values = _flatten_to_list(raw_array)
            except Exception:
                continue
            if len(values) != 1:
                continue
            scalars[node.output[0]] = attr.t
    return scalars


def _analyse_onnx(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "nms_in_graph": False,
        "providers": ["onnxruntime"],
    }
    try:
        import onnx  # type: ignore
    except Exception as exc:
        metadata["analysis_error"] = f"import:{exc}"
        return metadata

    try:
        model = onnx.load(str(path), load_external_data=False)  # type: ignore[attr-defined]
    except Exception as exc:
        metadata["analysis_error"] = f"load:{exc}"
        return metadata

    graph = getattr(model, "graph", None)
    if graph is None:
        metadata["analysis_error"] = "graph-missing"
        return metadata

    initializers = {init.name: init for init in getattr(graph, "initializer", [])}
    constants = _extract_constant_scalars(graph)

    def _resolve_scalar(name: str) -> Optional[float]:
        value = _resolve_scalar_from_initializer(initializers, name)
        if value is not None:
            return value
        tensor = constants.get(name)
        if tensor is None:
            return None
        return _resolve_scalar_from_initializer({"tmp": tensor}, "tmp")

    nms_node = None
    for node in getattr(graph, "node", []):
        if node.op_type.lower() in _NMS_OP_NAMES:
            nms_node = node
            break
        lowered = node.op_type.lower()
        if "nms" in lowered and node.op_type.isupper():
            nms_node = node
            break

    if nms_node is None:
        metadata["nms_in_graph"] = False
        return metadata

    metadata["nms_in_graph"] = True
    metadata["nms_op_type"] = nms_node.op_type
    inputs = list(getattr(nms_node, "input", []))

    max_det = None
    conf_thres = None
    if len(inputs) >= 3:
        max_det = _resolve_scalar(inputs[2])
    if len(inputs) >= 5:
        conf_thres = _resolve_scalar(inputs[4])

    if max_det is not None:
        try:
            metadata["max_det"] = int(round(max_det))
        except Exception:
            metadata["max_det"] = max_det
    if conf_thres is not None:
        metadata["within_graph_conf_thres"] = float(conf_thres)

    metadata["providers"] = ["onnxruntime", "onnxruntime_cuda"]
    if metadata["nms_in_graph"]:
        metadata["providers"].append("tensorrt")
        metadata["providers"].append("directml")

    return metadata


def _analyse_tensorrt(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "nms_in_graph": ".nms." in path.name.lower(),
        "providers": ["tensorrt"],
    }
    return metadata


def _analyse_torch(path: Path) -> Dict[str, Any]:
    return {
        "nms_in_graph": False,
        "providers": ["torch"],
    }


def analyse_artifact(path: Path) -> Dict[str, Any]:
    """
    Return a JSON-serialisable metadata dictionary describing *path*.
    """
    path = path.resolve()
    suffix = path.suffix.lower()
    metadata: Dict[str, Any] = {
        "path": path.name,
        "suffix": suffix,
        "size": path.stat().st_size if path.exists() else None,
        "analysed_at": _now_iso(),
        "nms_in_graph": False,
        "providers": [],
    }

    if suffix == ".onnx":
        metadata.update(_analyse_onnx(path))
    elif suffix in {".engine", ".plan"}:
        metadata.update(_analyse_tensorrt(path))
    elif suffix in {".pt", ".pth", ".torchscript"}:
        metadata.update(_analyse_torch(path))
    else:
        metadata["providers"] = []

    return metadata
