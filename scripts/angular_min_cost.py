"""Minimum-violation binary angular rounding, including infeasible assignments.

The native Gray-code search minimizes a sum of squared length violations over
all 2**n floor/ceiling choices. Pair areas use the same chord-corrected estimates
as the optimizer. A returned assignment is a proposal, not a validated packing.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from semicircle_packing.exact_area import polygon_intersection_areas, polygon_intersection_areas_and_jacobian
from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta

try:
    from .angular_rounder import _pair_indices
except ImportError:
    from angular_rounder import _pair_indices


@lru_cache(maxsize=1)
def _kernel():
    source=Path(__file__).with_name("_angular_min_cost.cpp")
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output=Path(tempfile.gettempdir())/f"packing-angular-cost-{digest}.so"
    if not output.exists():
        temporary=output.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(source),"-o",str(temporary)],check=True)
        temporary.replace(output)
    library=ctypes.CDLL(str(output));function=library.packing_min_binary_cost
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    function.argtypes=[ctypes.c_int,array,array,array,integer,array];function.restype=None
    return library,function


@lru_cache(maxsize=8)
def _geometry(n):return FastGeometry(n)


def _solve_tables(unary, pairs, preference=None):
    """Minimize nonnegative pairwise costs; angular preference breaks exact ties."""
    unary=np.ascontiguousarray(unary,dtype=np.float64)
    if unary.ndim!=2 or unary.shape[1]!=2 or not 1<=len(unary)<=20:
        raise ValueError("Expected unary costs of shape (n,2), 1 <= n <= 20")
    n=len(unary);pairs=np.ascontiguousarray(pairs,dtype=np.float64).reshape(n*(n-1)//2,2,2)
    if preference is None:preference=np.zeros_like(unary)
    preference=np.ascontiguousarray(preference,dtype=np.float64).reshape(n,2)
    if not np.isfinite(unary).all() or not np.isfinite(pairs).all() or not np.isfinite(preference).all():
        raise ValueError("Costs must be finite")
    if np.any(unary<0) or np.any(pairs<0):raise ValueError("Violation costs must be nonnegative")
    assignment=np.empty(n,dtype=np.int32);output=np.empty(3)
    _kernel()[1](n,unary,pairs,preference,assignment,output)
    return assignment,dict(energy=float(output[0]),preference=float(output[1]),enumerated=int(output[2]))


def minimize_angles(p, target, center=None, area_scale=None):
    """Return (six-decimal candidate, info), even when every assignment overlaps.

    area_scale optionally supplies the optimizer's n*(n-1)/2 area Jacobian norms.
    With no center, two alternating assignment/MEC updates are tried and the
    best proposal is returned at its own true MEC center. A supplied center is
    respected. Angular proximity only breaks exact violation-energy ties.
    """
    started=time.perf_counter();p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not 1<=len(p)<=20 or not np.isfinite(p).all():
        raise ValueError("Expected finite placements of shape (n,3), 1 <= n <= 20")
    if not np.isfinite(target) or target<=0:raise ValueError("Target radius must be positive")
    if np.max(np.abs(p[:,:2]-np.round(p[:,:2],6)))>1e-10:
        raise ValueError("Positions must already be on the six-decimal grid")
    n=len(p);pair_ids,indices=_pair_indices(n)
    theta=np.arctan2(np.sin(p[:,2]),np.cos(p[:,2]));units=theta*1e6;nearest=np.rint(units)
    units=np.where(np.abs(units-nearest)<1e-7,nearest,units)
    choices=np.c_[np.floor(units),np.ceil(units)]*1e-6
    choices=np.array([[canonicalize_theta(value)for value in row]for row in choices])
    variants=np.repeat(p,2,axis=0);variants[:,:2]=np.round(variants[:,:2],6);variants[:,2]=choices.ravel()
    areas=polygon_intersection_areas(variants)[indices]
    if area_scale is None:
        _,jac=polygon_intersection_areas_and_jacobian(p)
        area_scale=np.maximum(np.linalg.norm(jac,axis=1),1e-3)
    area_scale=np.asarray(area_scale,dtype=float)
    if area_scale.shape!=(n*(n-1)//2,) or not np.isfinite(area_scale).all() or np.any(area_scale<=0):
        raise ValueError("area_scale must contain one positive finite norm per pair")
    pair_violation=np.maximum(areas-1e-6,0)/area_scale[:,None,None]
    pair_cost=np.ascontiguousarray(pair_violation**2)
    angle_delta=choices-theta[:,None]
    preference=np.arctan2(np.sin(angle_delta),np.cos(angle_delta))**2
    supplied=center is not None
    current_center=np.asarray(compute_mec_fast(p)[:2] if center is None else center,dtype=float)
    if current_center.shape!=(2,) or not np.isfinite(current_center).all():raise ValueError("Center must contain two finite numbers")

    def unary_at(q):
        relative=variants.copy();relative[:,:2]-=q
        gaps,_=_geometry(2*n).boundary(np.r_[relative.ravel(),float(target)])
        return np.maximum(-gaps.reshape(2*n,3).min(axis=1),0).reshape(n,2)

    best=None;total_enumerated=0;seen=set();cost_build_seconds=time.perf_counter()-started
    for _ in range(1 if supplied else 3):
        unary_violation=unary_at(current_center)
        assignment,native=_solve_tables(unary_violation**2,pair_cost,preference)
        total_enumerated+=native["enumerated"]
        candidate=p.copy();candidate[:,:2]=np.round(candidate[:,:2],6);candidate[:,2]=choices[np.arange(n),assignment]
        actual_center=current_center if supplied else np.asarray(compute_mec_fast(candidate)[:2])
        actual_unary=unary_at(actual_center)[np.arange(n),assignment]
        selected_pair=pair_violation[np.arange(len(pair_ids)),assignment[pair_ids[:,0]],assignment[pair_ids[:,1]]]
        energy=float(actual_unary@actual_unary+selected_pair@selected_pair)
        tie=float(preference[np.arange(n),assignment].sum())
        if best is None or (energy,tie)<(best[0],best[1]):
            info=dict(center=actual_center.tolist(),energy=energy,max_violation=float(max(actual_unary.max(initial=0),selected_pair.max(initial=0))),
                overlap_pairs=int(np.count_nonzero(selected_pair)),containment_shapes=int(np.count_nonzero(actual_unary)),
                assignment=assignment.tolist(),feasible_estimate=bool(energy==0),preference=tie,fixed_center=supplied)
            best=(energy,tie,candidate.copy(),info)
        key=tuple(assignment)
        if key in seen or energy==0:break
        seen.add(key);current_center=actual_center
    candidate,info=best[2],best[3]
    info.update(enumerated=total_enumerated,cost_build_seconds=cost_build_seconds,seconds=time.perf_counter()-started)
    return candidate,info
