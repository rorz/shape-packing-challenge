"""Exact separating-axis constraints and analytic derivatives for fast searches.

Semicircle polygons are deliberately absent from the optimization loop. For a
unit direction u the support is 1 on the curved side and |u dot tangent| on the
flat side. The maximum separating gap has only eleven possible analytic axes:
two flat normals and nine circle/endpoint stationary directions. This computes
the exact signed separation, including a useful penetration depth for overlaps.

The tiny C++ kernel builds lazily into the OS temporary directory (requires a
C++ compiler); no compiled artifact is stored in the source tree.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np


class FastGeometry:
    """Constraints on z=[x0,y0,theta0,...,radius], feasible when g(z)>=0."""

    def __init__(self, n: int = 15):
        self.n = n
        self.n_pairs = n * (n - 1) // 2
        self.rows = self.n_pairs + n
        self.cols = 3 * n + 1
        source = Path(__file__).with_name("_fast_geometry.cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / f"semicircle-geometry-{digest}.so"
        if not target.exists():
            # An atomic rename permits independent search workers to initialize.
            import os
            temporary = target.with_suffix(f".{os.getpid()}.so")
            subprocess.run(
                ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(temporary)],
                check=True, capture_output=True,
            )
            temporary.replace(target)
        self._library = ctypes.CDLL(str(target))
        self._function = self._library.packing_constraints
        ptr = ctypes.POINTER(ctypes.c_double)
        self._function.argtypes = [ctypes.c_int, ptr, ptr, ptr]
        self._function.restype = None
        self._boundary_function = self._library.packing_boundary_constraints
        self._boundary_function.argtypes = [ctypes.c_int, ptr, ptr, ptr]
        self._boundary_function.restype = None
        self._last_z = None
        self._last_g = None
        self._last_jac = None

    def evaluate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return exact gaps and their dense analytic Jacobian."""
        z = np.ascontiguousarray(z, dtype=np.float64)
        if z.shape != (self.cols,):
            raise ValueError(f"Expected vector of shape {(self.cols,)}, got {z.shape}")
        if self._last_z is not None and np.array_equal(z, self._last_z):
            return self._last_g.copy(), self._last_jac.copy()
        gaps = np.empty(self.rows)
        jac = np.empty((self.rows, self.cols))
        ptr = ctypes.POINTER(ctypes.c_double)
        self._function(self.n, z.ctypes.data_as(ptr), gaps.ctypes.data_as(ptr), jac.ctypes.data_as(ptr))
        self._last_z, self._last_g, self._last_jac = z.copy(), gaps, jac
        return gaps.copy(), jac.copy()

    def constraints(self, z: np.ndarray) -> np.ndarray:
        return self.evaluate(z)[0]

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.evaluate(z)[1]

    def penalty(self, placements: np.ndarray, radius: float, margin: float = 0.0):
        """Return half squared violation, placement gradient, radius gradient.

        A positive margin enforces clearance. A negative margin permits a small
        penetration; its area must still be checked by the reference validator.
        """
        z = np.r_[np.asarray(placements).reshape(-1), radius]
        gaps, jac = self.evaluate(z)
        violations = np.minimum(gaps - margin, 0.0)
        gradient = violations @ jac
        return float(0.5 * violations.dot(violations)), gradient[:-1].reshape(self.n, 3), float(gradient[-1])

    def boundary(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 3n boundary inequalities, retaining both endpoint rows.

        Rows per shape are the negative-tangent endpoint, positive-tangent
        endpoint, and the radial arc extreme. The latter row is a constant 1
        when the radial direction is outside the arc and hence irrelevant.
        """
        z = np.ascontiguousarray(z, dtype=np.float64)
        if z.shape != (self.cols,):
            raise ValueError(f"Expected vector of shape {(self.cols,)}, got {z.shape}")
        gaps = np.empty(3*self.n)
        jac = np.empty((3*self.n, self.cols))
        ptr = ctypes.POINTER(ctypes.c_double)
        self._boundary_function(self.n, z.ctypes.data_as(ptr), gaps.ctypes.data_as(ptr), jac.ctypes.data_as(ptr))
        return gaps, jac

    def boundary_constraints(self, z: np.ndarray) -> np.ndarray:
        return self.boundary(z)[0]

    def boundary_jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.boundary(z)[1]
