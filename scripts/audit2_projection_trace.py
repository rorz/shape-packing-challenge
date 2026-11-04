"""Instrument one eight-step projection frame; this is not a search campaign."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from angular_min_cost import minimize_angles
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, load


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/2026-11-04/round2/audit'


def trace_frame(mincost,compression=False):
    source=ROOT/'scripts/experiment_position_projection.py'
    lines=source.read_text().splitlines()
    decoded=next(i+1 for i,line in enumerate(lines)if 'if not np.isfinite(n).all()'in line)
    update=next(i+1 for i,line in enumerate(lines)if 'x+=args.beta*'in line)
    rows=[]
    def tracer(frame,event,arg):
        if frame.f_code.co_filename!=str(source):return None
        if event=='line' and frame.f_lineno in (decoded,update):
            state=frame.f_locals
            if 'n' not in state:return tracer
            row=dict(stage='decoded'if frame.f_lineno==decoded else'update',iteration=int(state['iteration']),
                     max_integer_step=float(np.abs(state['n']).max()),x_norm=float(np.linalg.norm(state['x'])))
            if frame.f_lineno==update:
                row.update(lp_norm=float(np.linalg.norm(state['lp'])),
                    completion_distance=float(np.linalg.norm(state['lp']-state['x'])),
                    feedback_norm=float(np.linalg.norm(state['projection']-state['lp'])),
                    violation=float(-state['actual'].min()),
                    nonlinear_intercept_drift=float(np.linalg.norm(state['local_offset']-state['offset'])))
            rows.append(row)
        return tracer
    argv=[str(source),'--input','experiments/branch-3.141616-3.141592.json',
          '--output',str(OUT/('trace-mincost.json'if mincost else'trace-continuous.json')),
          '--seconds','5','--iterations','8','--max-restarts','0','--frame-angle','0',
          '--seed','202611046','--integer-boundary-angle']
    if mincost:argv.append('--mincost-projection')
    if compression:
        argv[argv.index('--input')+1]='tests/fixtures/packing_valid_20261104.json'
        argv[argv.index('--seed')+1]='110442'
        argv.extend(['--target','2.9485269238','--movement','1000','--beta','1.1',
                     '--integer-angle-indices','13'])
    previous=sys.argv;capture=io.StringIO()
    try:
        sys.argv=argv;sys.settrace(tracer)
        with contextlib.redirect_stdout(capture):runpy.run_path(str(source),run_name='__main__')
    finally:sys.settrace(None);sys.argv=previous
    return dict(command=argv,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),trace=rows,stdout=capture.getvalue())


def nonorthogonal_example():
    p=load('tests/fixtures/packing_valid_20261104.json');cx,cy,r=compute_mec_fast(p)
    centered=p.copy();centered[:,:2]-=[cx,cy];c=AreaConstraints(centered)
    g,j=c.evaluate(np.r_[centered.ravel(),r+1e-9]);J=j[g<1e-6,:45]
    U,_,_=np.linalg.svd(J,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
    C=-w[:,None]*np.c_[J[:,0::3].sum(1),J[:,1::3].sum(1)]
    theta_columns=np.arange(2,45,3);flat=np.zeros((2,15));flat[0,1]=.1;flat[0,5]=-.1;flat[1,6]=.1;flat[1,10]=-.1
    angle=np.vstack([np.eye(15)*1e-5,w[:,None]*J[:,theta_columns],flat])
    center=np.vstack([np.zeros((15,2)),C,np.zeros((2,2))])
    P=np.eye(len(angle))-center@np.linalg.pinv(center)
    fraction=.9;query=p.copy();query[14,2]+=fraction*1e-6
    completion,info=minimize_angles(query,r+1e-9)
    selected_delta=float((completion[14,2]-p[14,2])/1e-6)
    selected_distance=float(np.linalg.norm(P@angle[:,14]*(selected_delta-fraction)))
    closer_distance=float(np.linalg.norm(P@angle[:,14]*(1-fraction)))
    return dict(fraction=fraction,selected_grid_delta=selected_delta,
        selected_euclidean_distance=selected_distance,closer_grid_distance=closer_distance,
        distance_ratio=selected_distance/closer_distance,completion_energy=info['energy'],
        interpretation='Minimum-violation completion is deliberately not the orthogonal lattice projection used in the Douglas-Rachford formula.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(continuous=trace_frame(False),mincost=trace_frame(True),
                logged_compression=trace_frame(True,compression=True),nonorthogonal=nonorthogonal_example())
    (OUT/'projection-trace.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
