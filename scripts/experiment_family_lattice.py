"""Search integer points near curved continuous-family sections, not one tangent."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.polish import AreaConstraints, load
from semicircle_packing.fast_mec import compute_mec_fast


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input',default='experiments/branch-3.141616-3.141592.json')
    pa.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/family'))
    pa.add_argument('--seed',type=int,default=110420)
    pa.add_argument('--count',type=int,default=16)
    pa.add_argument('--seconds-per-seed',type=float,default=15)
    pa.add_argument('--target',type=float,default=2.948526945)
    pa.add_argument('--amplitude',type=float,default=.001)
    args=pa.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    p=load(args.input);cx,cy,R=compute_mec_fast(p);p[:,:2]-=[cx,cy]
    c=AreaConstraints(p);g,j=c.evaluate(np.r_[p.ravel(),R]);active=g<1e-7
    _,singular,V=np.linalg.svd(j[active,:45],full_matrices=True)
    B=V[40:]
    fixed=np.zeros((3,45));fixed[0,5]=1;fixed[0,17]=-1;fixed[1,20]=1;fixed[1,32]=-1;fixed[2,2]=1
    _,_,T=np.linalg.svd(fixed@B.T,full_matrices=True);soft=T[3:]@B
    E=np.column_stack([np.vstack([fixed,soft]),np.zeros(5)])
    objective=np.zeros(46);objective[-1]=1;baseline=E@np.r_[p.ravel(),R]
    rng=np.random.default_rng(args.seed);started=time.monotonic();records=[]
    for index in range(args.count):
        angle=2*np.pi*index/args.count
        amplitude=args.amplitude*(.5 if index%2 else 1.)
        alpha=amplitude*np.array([np.cos(angle),np.sin(angle)])
        target=baseline.copy();target[-2:]+=alpha
        initial=np.r_[(p+(alpha@soft).reshape(15,3)).ravel(),R+1e-7]
        result=minimize(lambda z:z[-1],initial,jac=lambda z:objective,method='SLSQP',
            constraints=[dict(type='ineq',fun=c.fun,jac=c.jac),dict(type='eq',fun=lambda z:E@z-target,jac=lambda z:E)],
            options=dict(ftol=2e-13,maxiter=500))
        violation=-float(c.fun(result.x).min());radius=float(result.x[-1]);q=result.x[:-1].reshape(15,3)
        path=args.output/f'seed-{index:02}.json';path.write_text(json.dumps([dict(zip(('x','y','theta'),map(float,row))) for row in q],indent=2)+'\n')
        record=dict(index=index,alpha=alpha.tolist(),radius=radius,violation=violation,success=bool(result.success),
                    singular_values=singular[-3:].tolist(),seconds=time.monotonic()-started)
        if violation<1e-9 and radius<args.target-1e-7:
            command=[sys.executable,'scripts/experiment_lll.py','--input',str(path),'--keep-seed','--seconds',str(args.seconds_per_seed),
                '--seed',str(args.seed+index),'--target',str(args.target),'--goal','2.9485269238','--incumbent','solution.json',
                '--movement','50','--concentration','10','--boundary-bias','5','--block-size','20',
                '--output',str(args.output/f'best-{index:02}.json'),'--checkpoint-prefix',f'seed-{index:02}']
            record['command']=command
            with (args.output/f'run-{index:02}.log').open('w') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.seconds_per_seed+20)
            record['exit_code']=process.returncode
            # Every child only writes a validated improvement; a winning child stops this campaign.
            for line in (args.output/f'run-{index:02}.log').read_text().splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if 'best' in event:record['best']=min(record.get('best',float('inf')),event['best'])
        records.append(record);(args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if record.get('best',float('inf'))<2.9485269238:break


if __name__=='__main__':main()
