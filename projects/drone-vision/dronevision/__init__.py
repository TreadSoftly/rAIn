"""
Public package namespace.
Importing the package sets the version and re-exports the public modules.
"""
from importlib.metadata import version as _v

__all__ = ["cli", "heatmap", "geo_sink", "predict_mp4", "lambda_like"]
__version__ = _v(__name__)
