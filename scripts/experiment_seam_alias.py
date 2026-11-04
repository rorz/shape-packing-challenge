"""Search the two additional scored angle values reachable only through a wrap.

Raw ±9.424778 scores as ∓3.141593. Ordinary raw ±3.141593 instead scores
as ∓3.141592. Fix the endpoint-sensitive angle to each extra state, vary the
curved manifold, and decode the remaining42 integer variables.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from experiment_mirror_family import Manifold,FLATS,ENDPOINT,wrap
from experiment_tied_batch import Frame
from experiment_batch_lattice import NativeBabai
from breakout_geometry import evaluate_batch
from semicircle_packing.polish import load,save,checked
from semicircle_packing.geometry import canonicalize_theta
from arena import check_solution


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='experiments/2026-11-04/round2/topology/continuous-best.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--seed',type=int,default=110479);ap.add_argument('--sigma',type=float,default=.00035);ap.add_argument('--target',type=float,default=2.94852698);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--block-size',type=int,default=20);ap.add_argument('--movement',type=float,default=50);ap.add_argument('--concentration',type=float,default=10);ap.add_argument('--boundary-bias',type=float,default=2);ap.add_argument('--regularization',type=float,default=1e-5);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(a.seed);model=Manifold(load(a.input));decoder=NativeBabai();decoder.reset();best=checked(load('solution.json')).score;elite=best+1e-7;nearest=float('inf');last=start
    a.pin_piece=ENDPOINT
    counts=dict(sections=0,frames=0,proposals=0,unique=0,rejected_stress=0,rejected_geometry=0,rejected_branch=0,checks=0,valid=0)
    def emit(event,**data):
        row=dict(event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row);print(line,flush=True)
        with (a.output/'events.jsonl').open('a')as f:f.write(line+'\n')
    for sign in (-1,1):
        raw=-sign*9.424778;geometry=sign*3.141593
        assert canonicalize_theta(raw)==geometry
        assert canonicalize_theta(geometry)!=geometry
    emit('start',alias_api_evidence='experiments/api_angle_probes.json',dimension=42,target=a.target,sigma=a.sigma)
    while time.monotonic()-start<a.seconds:
        sign=(-1,1)[counts['sections']%2];angle_value=sign*3.141593;a.pin_raw=-sign*9.424778;alpha=rng.normal(0,a.sigma,2)
        # Determine the physical wrap cell, then solve matching exact raw D.
        initial=model.p.copy();phi=angle_value-initial[ENDPOINT,2];theta=wrap(initial[:,2]+phi)
        raw=np.round([theta[i]-theta[j]for i,j in FLATS],6)
        raw[0]+=rng.integers(-2,3)*1e-6;raw[1]+=rng.integers(-1,2)*1e-6
        p,metadata=model.section(raw.tolist(),alpha);counts['sections']+=1
        if metadata['violation']>2e-9 or metadata['radius']>a.target-1e-7:counts['rejected_geometry']+=1;continue
        phi=angle_value-p[ENDPOINT,2];cs,sn=np.cos(phi),np.sin(phi);posed=p.copy();posed[:,:2]=p[:,:2]@np.array([[cs,sn],[-sn,cs]]);posed[:,2]=wrap(p[:,2]+phi);posed[ENDPOINT,2]=angle_value
        actual_D=np.array([posed[i,2]-posed[j,2]for i,j in FLATS])
        if np.max(abs(actual_D-np.round(actual_D,6)))>2e-11:counts['rejected_branch']+=1;continue
        try:frame=Frame(posed,metadata['radius'],FLATS,a)
        except ValueError:counts['rejected_stress']+=1;continue
        geometry,units,coeff,ambient,patches=frame.sample(rng,a.batch,a,decoder);counts['frames']+=1;counts['proposals']+=len(geometry)
        if counts['frames']<=2:
            expected=np.array([frame.M.babai(v.tolist())for v in ambient[:128]])
            assert np.array_equal(expected,coeff[:128])
            for k in range(16):
                submitted=frame.raw(coeff[k]);canonical=np.round(submitted,6);canonical[:,2]=[canonicalize_theta(t)for t in canonical[:,2]]
                assert np.max(abs(canonical-geometry[k]))<5e-15
            emit('decoder_audit',coefficients=128,raw_geometry_checks=16,pin_geometry=angle_value,pin_raw=a.pin_raw)
        keep,_=decoder.unique(units);geometry=geometry[keep];coeff=coeff[keep]
        if not len(geometry):continue
        areas,circles,ok=evaluate_batch(geometry,centers=[0.,0.]);counts['unique']+=len(geometry);violations=np.maximum(np.max((areas-1e-6)/frame.c.scale,axis=1),circles[:,2]-a.target);index=int(np.argmin(violations))
        if violations[index]<nearest:
            nearest=float(violations[index]);submitted=frame.raw(coeff[index]);save(a.output/'nearest-raw.json',submitted);np.savez(a.output/'nearest.npz',placements=submitted,geometry=geometry[index],origin=posed,moves=frame.moves.reshape(42,15,3),alpha=alpha,branch=raw,center=circles[index,:2]);emit('nearest',violation=nearest,radius=float(circles[index,2]),branch=raw.tolist(),alpha=alpha.tolist())
        eligible=np.flatnonzero((areas.max(1)<=1e-6+1e-11)&(circles[:,2]<elite-2e-12))
        for index in eligible[np.argsort(circles[eligible,2])]:
            submitted=frame.raw(coeff[index]);v=checked(submitted);counts['checks']+=1
            if not v.valid:continue
            counts['valid']+=1;elite=min(elite,v.score);save(a.output/f"elite-{counts['valid']:03d}.json",submitted);emit('valid',score=v.score,raw_pin=a.pin_raw)
            if v.score<best:
                save(a.output/'candidate.json',submitted);response=check_solution(json.loads((a.output/'candidate.json').read_text()));(a.output/'candidate-api.json').write_text(json.dumps(response,indent=2)+'\n');emit('api',**response)
                if response.get('valid')and response['score']<best:best=response['score'];save(a.output/'best.json',submitted)
        if best<2.9485269238856113:break
        if time.monotonic()-last>10:emit('progress',best=best,nearest=nearest,counts=counts.copy());last=time.monotonic()
    result=dict(best=best,nearest=nearest,counts=counts,goal_met=best<2.9485269238856113);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**result)


if __name__=='__main__':main()
