"""Finite pose domains covering every subgrid translation of one physical frame.

Each output row is *scored* six-decimal geometry. The independent per-piece
unions can combine choices from different translation phases; this intentionally
enlarges the search space. It does not invalidate an accepted packing, provided
the caller still checks its overlaps and enclosing circle.
"""
from __future__ import annotations

import hashlib
import itertools
import operator

import numpy as np

from semicircle_packing.geometry import canonicalize_theta

MICRO = 1e-6
HALF_GRID = 0.5e-6


def _width(value, name):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def translation_axis_bounds(value, xy_width=1):
    """Inclusive integer-micro bounds for one coordinate's complete phase union.

    Floating addition, NumPy decimal rounding, and adding a fixed offset are
    monotone on this finite range. Therefore rounding the two phase endpoints,
    followed by the two extreme local offsets, bounds every intervening choice.
    One outward representable step at each operation conservatively includes
    ties and endpoint arithmetic. At generic coordinates the result has four
    values for width one; exact grid coordinates can have five.
    """
    width = _width(xy_width, "xy_width")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Coordinates must be finite")
    if abs(value) * 1e6 + width + 4 >= 2 ** 52:
        raise ValueError("Coordinate and width exceed exact integer-micro range")
    low = np.nextafter(value - HALF_GRID, -np.inf)
    high = np.nextafter(value + HALF_GRID, np.inf)
    low_base, high_base = np.round([low, high], 6)
    low_pose = np.round(np.nextafter(low_base - width * MICRO, -np.inf), 6)
    high_pose = np.round(np.nextafter(high_base + width * MICRO, np.inf), 6)
    return int(np.rint(low_pose * 1e6)), int(np.rint(high_pose * 1e6))


def _raw_encoding(scored_theta):
    if canonicalize_theta(scored_theta) == scored_theta:
        return scored_theta
    for alias in (-9.424778, 9.424778):
        if canonicalize_theta(alias) == scored_theta:
            return alias
    raise ValueError(f"No raw encoding for scored angle {scored_theta}")


def _angles(physical_theta, angle_width):
    # This is the production generator's scored-base neighborhood for stable
    # angles. The exceptional ±3pi aliases instead retain their once-scored
    # endpoint through an inverse encoding, avoiding a second normalization.
    scored = canonicalize_theta(float(np.round(physical_theta, 6)))
    raw = _raw_encoding(scored)
    values = {round(scored * 1e6)}
    for offset in range(-angle_width, angle_width + 1):
        submitted = float(np.round(raw + offset * MICRO, 6))
        values.add(round(canonicalize_theta(submitted) * 1e6))
    return scored, sorted(values)


def make_translation_domains(physical, xy_width=1, angle_width=1, max_poses=2000):
    """Return scored poses covering all common XY phases in [-0.5, 0.5] micro.

    For ordinary stable angles this contains every pose produced by
    ``make_domains(scored_once(physical + shift), xy_width, angle_width)``.
    The output also supports the exceptional scored angles ±3.141593 through
    raw ±3pi aliases, which the old generator does not preserve. Callers must encode those
    rows as raw input before using a validator that canonicalizes its input.

    The zero-phase rounded frame comes first. A conservative size check occurs
    before allocating any pose products; oversized requests raise ValueError,
    never silently truncate. Piece labels and orientation are not quotiented.
    """
    xy_width = _width(xy_width, "xy_width")
    angle_width = _width(angle_width, "angle_width")
    max_poses = _width(max_poses, "max_poses")
    if max_poses == 0:
        raise ValueError("max_poses must be positive")
    p = np.asarray(physical, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("physical must be a nonempty array of finite (x, y, theta) rows")

    bounds = []
    for index, row in enumerate(p):
        xb = translation_axis_bounds(row[0], xy_width)
        yb = translation_axis_bounds(row[1], xy_width)
        nominal = (xb[1] - xb[0] + 1) * (yb[1] - yb[0] + 1) * (2 * angle_width + 1)
        if nominal > max_poses:
            raise ValueError(f"Piece {index} needs at most {nominal} poses, exceeding max_poses={max_poses}")
        bounds.append((xb, yb))

    domains = []
    for row, (xb, yb) in zip(p, bounds):
        scored_theta, angles = _angles(row[2], angle_width)
        base = (int(np.rint(np.round(row[0], 6) * 1e6)),
                int(np.rint(np.round(row[1], 6) * 1e6)), round(scored_theta * 1e6))
        poses = [base]
        poses.extend(pose for pose in itertools.product(range(xb[0], xb[1] + 1),
                                                       range(yb[0], yb[1] + 1), angles)
                     if pose != base)
        domain = np.asarray(poses, dtype=np.float64) / 1e6
        domains.append(domain)
    return domains


def domain_digest(domains):
    """Hash complete integer domains independently of row order or duplicates."""
    digest = hashlib.sha256()
    digest.update(len(domains).to_bytes(8, "little"))
    for domain in domains:
        values = np.asarray(domain, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("Each domain must contain finite (x, y, theta) rows")
        rows = np.unique(np.rint(values * 1e6).astype("<i8"), axis=0)
        digest.update(len(rows).to_bytes(8, "little"))
        digest.update(rows.tobytes())
    return digest.hexdigest()
