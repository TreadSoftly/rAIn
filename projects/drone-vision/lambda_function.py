"""Compat shim so local tests can `import lambda_function`."""

from __future__ import annotations
import importlib
import os
import pathlib
import sys
import types

# --------------------------------------------------------------------------- #
# 1.  Test-friendly environment defaults                                       #
# --------------------------------------------------------------------------- #
os.environ.setdefault("OUT_BUCKET", "out")   # mocked S3 bucket used in tests

# --------------------------------------------------------------------------- #
# 2.  Ensure the *local* lambda package shadows any PyPI package named lambda #
# --------------------------------------------------------------------------- #
repo_root = pathlib.Path(__file__).resolve().parent
lambda_dir = repo_root / "lambda"

# a) put the repo root on sys.path so `import lambda.app` can be found
sys.path.insert(0, str(repo_root))

# b) if the folder has no __init__.py, create a dummy package object
if (lambda_dir / "__init__.py").exists() is False:
    pkg = types.ModuleType("lambda")
    pkg.__path__ = [str(lambda_dir)]
    sys.modules["lambda"] = pkg

# --------------------------------------------------------------------------- #
# 3.  Re-export everything from lambda.app                                     #
# --------------------------------------------------------------------------- #
_app = importlib.import_module("lambda.app")
globals().update(_app.__dict__)   # so `lambda_function.foo` mirrors `app.foo`
app = _app                        # what the tests expect
