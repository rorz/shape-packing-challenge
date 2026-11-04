"""Enumerate the orientations that can change an angular family's MEC.

This is a geometric center oracle, not an overlap validator. Unenumerated
orientations must be certified radial or interior; returned center/radius
invariance is also checked directly. Limits are explicit in the returned info.
"""
from __future__ import annotations

import itertools
import math
import numpy as np

from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle, canonicalize_theta, farthest_boundary_point_from


def _extents(piece,angles,center):
    x,y=piece[:2]-center;sn,co=np.sin(angles),np.cos(angles)
    endpoints=np.maximum(np.hypot(x+sn,y-co),np.hypot(x-sn,y+co))
    return np.maximum(endpoints,np.where(x*co+y*sn>=0,np.hypot(x,y)+1,0))


def boundary_centers(p,angle_choices=None,target=None,*,max_combinations=256,tolerance=5e-12):
    """Return (center records, info) for fixed positions and finite angle choices.

    angle_choices may contain two or many physical angles per piece. If omitted,
    normalized floor/ceiling grid angles are used. Records contain center,
    radius, fixed_angles, and verified_invariant. A complete_geometry flag is
    meaningful only for this finite family and the stated numerical tolerance.
    Pairwise overlap feasibility remains a separate problem.
    """
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():raise ValueError('Expected finite (n,3) placements')
    if max_combinations<1:raise ValueError('max_combinations must be positive')
    if angle_choices is None:
        theta=np.arctan2(np.sin(p[:,2]),np.cos(p[:,2]))*1e6
        angle_choices=[[canonicalize_theta(v*1e-6)for v in (np.floor(t),np.ceil(t))]for t in theta]
    if len(angle_choices)!=len(p):raise ValueError('Expected one angle-choice array per piece')
    choices=[]
    for values in angle_choices:
        values=np.unique(np.asarray(values,dtype=float).ravel())
        if not len(values) or not np.isfinite(values).all():raise ValueError('Angle choices must be finite and nonempty')
        choices.append(values)
    reference=np.asarray(compute_mec_fast(p));center=reference[:2];radius=float(reference[2])
    delta=max(float(np.max(2*np.abs(np.sin(np.arctan2(np.sin(v-p[i,2]),np.cos(v-p[i,2]))/2))))for i,v in enumerate(choices))+1e-12
    # If A and B differ by Hausdorff distance delta, B's MEC center encloses A
    # at radius at most R_A+2delta. Supporting points of A imply this bound.
    center_bound=2*math.sqrt(radius*delta+delta*delta)+tolerance
    endpoint=[];classification=[]
    for i,values in enumerate(choices):
        normals=np.c_[np.cos(values),np.sin(values)]
        radial=bool(np.min(normals@(p[i,:2]-center))>=center_bound+tolerance)
        interior=bool(_extents(p[i],values,center).max()+center_bound<radius-delta-tolerance)
        classification.append('radial'if radial else'interior'if interior else'endpoint_sensitive')
        if not(radial or interior):endpoint.append(i)
    total=math.prod(len(choices[i])for i in endpoint)
    records=[];checked=0;verified=0;excluded=0
    options=[sorted(choices[i],key=lambda a:abs(math.atan2(math.sin(a-p[i,2]),math.cos(a-p[i,2]))))for i in endpoint]
    for selected in itertools.islice(itertools.product(*options),max_combinations):
        representative=p.copy()
        for i,values in enumerate(choices):representative[i,2]=values[0]
        fixed={i:float(value)for i,value in zip(endpoint,selected)}
        for i,value in fixed.items():representative[i,2]=value
        cx,cy,R=compute_mec_fast(representative);query=np.array([cx,cy]);checked+=1
        invariant=True
        for i,values in enumerate(choices):
            if i in fixed:continue
            if _extents(p[i],values,query).max()>R+tolerance:invariant=False;break
            far=np.array(farthest_boundary_point_from(Semicircle(*representative[i]),cx,cy))
            if R-np.linalg.norm(far-query)<=tolerance:
                # Every actual support witness must remain a point of every
                # unenumerated orientation, not merely fit its enclosing disk.
                witness=far-p[i,:2]
                if np.min(np.c_[np.cos(values),np.sin(values)]@witness)<-tolerance:
                    invariant=False;break
        verified+=int(invariant)
        record=dict(center=[cx,cy],radius=R,fixed_angles=fixed,verified_invariant=invariant)
        if target is not None and invariant and R>target+tolerance:excluded+=1
        else:records.append(record)
    info=dict(endpoint_indices=endpoint,classification=classification,hausdorff_bound=delta,
              center_shift_bound=center_bound,total_combinations=total,tested=checked,
              verified_combinations=verified,excluded_by_radius=excluded,
              truncated=checked<total,complete_geometry=bool(checked==total and verified==checked),
              tolerance=tolerance)
    return records,info
