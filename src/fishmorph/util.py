from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np


def clean(value):
    """Strict JSON: missing numeric values are null, never non-standard NaN tokens."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value, indent=None):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False, indent=indent)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def environment():
    result = {}
    for package in ("numpy", "scipy", "Pillow", "scikit-image", "pydantic", "fastapi", "uvicorn", "trimesh", "statsmodels"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    try:
        import cv2
        result["opencv"] = cv2.__version__
    except ImportError:
        result["opencv"] = "not installed"
    return result


def code_fingerprint():
    """Hash shipped Python implementation; local edits invalidate derived caches."""
    root = Path(__file__).parent
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*.py"))}
    from .anatomy import LANDMARKS, RATIOS, FIN_LANDMARKS
    return digest({"files": files, "landmarks": LANDMARKS, "ratios": RATIOS,
                   "fin_landmarks": FIN_LANDMARKS})


def protocol_fingerprint():
    """Identity of definitions, separate from implementation/cache identity."""
    from .anatomy import LANDMARKS, RATIOS, FIN_LANDMARKS
    from . import SCHEMA_VERSION
    return digest({"landmarks": LANDMARKS, "ratios": RATIOS,
                   "fin_landmarks": FIN_LANDMARKS, "schema": SCHEMA_VERSION})
