"""Analytic semicircle intersection areas for fast tolerance-aware polishing.

Green's theorem integrates the arcs and line segments inside the other shape.
The official polygon validator remains the final authority for saved candidates.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

_kernel = None


def _load():
    global _kernel
    if _kernel is None:
        source = Path(__file__).with_name("area_kernel.cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / f"packing-area-{digest}.dylib"
        if not target.exists():
            temporary = target.with_suffix(f".{__import__('os').getpid()}.tmp")
            subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC",
                            str(source), "-o", str(temporary)], check=True)
            temporary.replace(target)
        _kernel = ctypes.CDLL(str(target))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        _kernel.packing_areas.argtypes = [ctypes.c_int, array, array]
        _kernel.packing_areas.restype = None
        _kernel.packing_areas_jacobian.argtypes = [ctypes.c_int, array, array, array]
        _kernel.packing_areas_jacobian.restype = None
        _kernel.packing_polygon_areas.argtypes = [ctypes.c_int, array, array]
        _kernel.packing_polygon_areas.restype = None
        _kernel.packing_polygon_areas_jacobian.argtypes = [ctypes.c_int, array, array, array]
        _kernel.packing_polygon_areas_jacobian.restype = None
    return _kernel


def intersection_areas(placements: np.ndarray) -> np.ndarray:
    """Return all i<j intersection areas in lexicographic pair order."""
    p = np.ascontiguousarray(placements, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(p) * (len(p) - 1) // 2)
    _load().packing_areas(len(p), p, out)
    return out


def intersection_areas_and_jacobian(placements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pair areas and analytic derivatives w.r.t. flattened placements.

    Boundary flux gives translation derivatives from the included arcs and flat
    segments. Rotation contributes only on a flat segment. Coincident circles
    use symmetric differences where shared boundaries make gradients singular.
    """
    p = np.ascontiguousarray(placements, dtype=np.float64).reshape(-1, 3)
    rows = len(p) * (len(p) - 1) // 2
    out = np.empty(rows)
    jac = np.empty((rows, len(p) * 3))
    _load().packing_areas_jacobian(len(p), p, out, jac)
    return out, jac


def polygon_intersection_areas(placements: np.ndarray) -> np.ndarray:
    """Estimate 4096-point polygon areas via an analytic chord correction.

    The first-order boundary correction integrates each arc's chord deficit.
    Errors at contacts are typically below 1e-12 area; the reference polygon
    validator remains authoritative. Coincident disks use the circle limit.
    """
    p = np.ascontiguousarray(placements, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(p)*(len(p)-1)//2)
    _load().packing_polygon_areas(len(p), p, out)
    return out


def polygon_intersection_areas_and_jacobian(placements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return chord-corrected areas with their analytic derivatives."""
    p = np.ascontiguousarray(placements, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(p)*(len(p)-1)//2)
    jac = np.empty((len(out), len(p)*3))
    _load().packing_polygon_areas_jacobian(len(p), p, out, jac)
    return out, jac
