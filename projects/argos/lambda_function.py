
"""Compat shim so local unit‑tests can simply `import lambda_function`.

It transparently re‑exports every name from `lambda.app`, ensuring the
repository's **local** `lambda/` package shadows any similarly‑named
distribution that might be installed from PyPI.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

# ────────────────────────────────────────────────────────────────
# 1.  Sensible defaults for *local* execution
# ────────────────────────────────────────────────────────────────
# moto‑mocked S3 in the test‑suite expects this bucket:
os.environ.setdefault("GEO_BUCKET", "out")   # ← match lambda.app default

# ────────────────────────────────────────────────────────────────
# 2.  Ensure *our* lambda package wins import‑resolution
# ────────────────────────────────────────────────────────────────
REPO_ROOT  = pathlib.Path(__file__).resolve().parent
LAMBDA_DIR = REPO_ROOT / "lambda"

# a) repo root must be importable so that `import lambda.app` resolves here
sys.path.insert(0, str(REPO_ROOT))

# b) create a runtime‑only namespace package if __init__.py is missing
if not (LAMBDA_DIR / "__init__.py").exists():
    pkg = types.ModuleType("lambda")
    pkg.__path__ = [str(LAMBDA_DIR)]
    sys.modules["lambda"] = pkg

# ────────────────────────────────────────────────────────────────
# 3.  Re‑export everything from lambda.app
# ────────────────────────────────────────────────────────────────
_app = importlib.import_module("lambda.app")
globals().update(_app.__dict__)   # so `lambda_function.foo` mirrors `app.foo`
app = _app                        # what AWS/SAM & unit‑tests expect
