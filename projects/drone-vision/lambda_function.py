"""Compat shim so *local* unit-tests can simply `import lambda_function`.

It transparently re-exports every name from `lambda.app`, while ensuring the
repository's **local** `lambda/` package shadows any similarly-named package
that might be installed from PyPI.

Nothing here should need changing during normal development - it is only a
thin wrapper so that tests and IDEs work the same way the AWS Lambda runtime
does.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Sensible environment defaults for *local* execution
# ──────────────────────────────────────────────────────────────────────────────
# The unit-tests spin up a moto-mocked S3 stack that expects this bucket.
os.environ.setdefault("OUT_BUCKET", "out")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Ensure *our* lambda package wins import-resolution
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent
LAMBDA_DIR = REPO_ROOT / "lambda"

# a)  Make repo root importable (so `import lambda.app` resolves locally)
sys.path.insert(0, str(REPO_ROOT))

# b)  If the folder lacks an __init__.py, create a runtime-only package object
if not (LAMBDA_DIR / "__init__.py").exists():
    pkg = types.ModuleType("lambda")
    pkg.__path__ = [str(LAMBDA_DIR)]
    sys.modules["lambda"] = pkg

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Re-export everything from lambda.app
# ──────────────────────────────────────────────────────────────────────────────
_app = importlib.import_module("lambda.app")
globals().update(_app.__dict__)   # so `lambda_function.foo` mirrors `app.foo`
app = _app                        # what AWS / SAM & tests expect
