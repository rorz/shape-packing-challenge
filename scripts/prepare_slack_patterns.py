"""Extract rotation-invariant weighted-clearance patterns from frozen witnesses."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints,load,checked
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from experiment_tied_batch import tied_transform


def simplex_projection(values,total):
    ordered=np.sort(values)[::-1];level=(np.cumsum(ordered)-total)/np.arange(1,len(values)+1)
    rho=np.flatnonzero(ordered>level)[-1]
    return np.maximum(values-level[rho],0)


def extract(origin,geometry):
    origin=origin.copy();cx,cy,R=compute_mec_fast(origin);origin[:,:2]-=[cx,cy]
    c=AreaConstraints(origin);g,J=c.evaluate(np.r_[origin.ravel(),R]);ids=np.flatnonzero(g<1e-7)
    _,T=tied_transform(origin,[(1,14),(4,8)])
    U,_,_=np.linalg.svd(J[ids,:45]@T,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
    if w.min()<=0:raise ValueError('Nonpositive constrained stress')
    posed=geometry.copy();cx,cy,r=compute_mec_fast(posed);posed[:,:2]-=[cx,cy]
    weighted=w*c.fun(np.r_[posed.ravel(),r])[ids]
    if weighted.sum()<=0:raise ValueError('No positive weighted clearance budget')
    pattern=simplex_projection(weighted,float(weighted.sum()));pattern/=pattern.sum()
    return pattern,weighted,ids,float(r)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path('experiments/2026-11-04/round3/mirror'))
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    sources=[(p,p.with_suffix('.npz'))for p in sorted(a.root.glob('*/elite-*.json'))if p.with_suffix('.npz').exists()]
    sources.extend((a.root/name/'nearest-raw.json',a.root/name/'nearest-state.npz')for name in ['neighbor-g','broad-l'])
    sources.append((Path('experiments/2026-11-04/round3/candidate-2.948526962953499.json'),a.root/'campaign-b/mirror-valid-state.npz'))
    patterns=[];weighted=[];scores=[];hashes=[];paths=[];labels=[];records=[];skipped=[];seen=set();active_ids=None;rotation_errors=[]
    for path,state in sources:
        if not(path.exists()and state.exists()):skipped.append(dict(path=str(path),reason='missing sidecar'));continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);raw=load(path)
        with np.load(state)as z:
            if np.max(abs(z['placements']-raw))>1e-12:
                skipped.append(dict(path=str(path),reason='sidecar/raw mismatch'));continue
            origin=z['origin'].copy()
        geometry=np.round(raw,6);geometry[:,2]=[canonicalize_theta(t)for t in geometry[:,2]]
        try:pattern,wg,ids,r=extract(origin,geometry)
        except ValueError as e:skipped.append(dict(path=str(path),reason=str(e)));continue
        if active_ids is None:active_ids=ids
        if len(ids)!=41 or not np.array_equal(active_ids,ids):
            skipped.append(dict(path=str(path),reason='different active constraint IDs'));continue
        validation=checked(raw);label='valid'if validation.valid else 'near_invalid'
        if len(patterns)<12:
            for angle in [.37,-1.083]:
                co,si=np.cos(angle),np.sin(angle);matrix=np.array([[co,si],[-si,co]])
                oo=origin.copy();qq=geometry.copy()
                for q in (oo,qq):q[:,:2]=q[:,:2]@matrix;q[:,2]+=angle
                pp,_,ii,_=extract(oo,qq)
                assert np.array_equal(ii,ids)
                rotation_errors.append(float(np.max(abs(pp-pattern))))
        patterns.append(pattern);weighted.append(wg);scores.append(float(validation.score)if validation.valid else r);hashes.append(digest);paths.append(str(path));labels.append(label)
        records.append(dict(path=str(path),sha256=digest,sidecar=str(state),sidecar_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),label=label,radius=scores[-1],minimum_weighted_slack=float(wg.min()),weighted_budget=float(wg.sum())))
    if not patterns:raise ValueError('No patterns extracted')
    assert max(rotation_errors)<1e-6
    np.savez_compressed(a.output/'patterns.npz',patterns=np.asarray(patterns),active_ids=active_ids,
        source_scores=np.asarray(scores),source_hashes=np.asarray(hashes),source_paths=np.asarray(paths),
        source_labels=np.asarray(labels),raw_weighted_g=np.asarray(weighted))
    report=dict(patterns=len(patterns),valid=labels.count('valid'),near_invalid=labels.count('near_invalid'),
        active_ids=active_ids.tolist(),rigid_rotation_checks=len(rotation_errors),maximum_rotation_pattern_error=max(rotation_errors),
        minimum_score=min(scores),maximum_row_sum_error=float(np.max(abs(np.asarray(patterns).sum(1)-1))),
        seconds=time.monotonic()-started,records=records,skipped=skipped,
        method='J@T constrained stress; actual free MEC; Euclidean simplex projection at preserved weighted budget; no extra quantization on rigid-rotation audit')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('records','skipped')}|dict(skipped=len(skipped))),flush=True)


if __name__=='__main__':main()
