"""Native angular completion on exact square-lattice symmetries of an incumbent."""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,json,time,shlex,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from angular_rounder import solve_binary_constraints
from experiment_neutral_network import polygon,radii_at
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import load,checked,save
from semicircle_packing.fast_mec import compute_mec_fast


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='solution.json');ap.add_argument('--output',default='experiments/2026-11-04/round2/neutral/orbits');ap.add_argument('--relay',type=float,default=3e-7);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();p=load(a.input);base=checked(p).score;cap=base+a.relay;records=[]
 log=out/'steps.md';log.write_text('# Exact-square orbit starts\n\nExact 90-degree coordinate swaps/reflections retain the x/y grid. Angle floor/ceil states are tested by native GEOS pair tables, not nearest independent rounding. Piece14 endpoint variants receive a free optimized center with the other14 full disks enclosed. The binary CSP covers all32768 angular hybrids at each center.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 for reflect in [False,True]:
  for quarter in range(4):
   q=p.copy()
   if reflect:q[:,1]*=-1;q[:,2]*=-1
   for _ in range(quarter):q[:,:2]=q[:,:2][:,[1,0]]*[-1,1]
   continuous=np.arctan2(np.sin(q[:,2]+quarter*np.pi/2),np.cos(q[:,2]+quarter*np.pi/2));units=continuous*1e6;nearest=np.rint(units);units=np.where(abs(units-nearest)<1e-7,nearest,units);choices=np.c_[np.floor(units),np.ceil(units)]*1e-6;choices=np.array([[canonicalize_theta(v)for v in row]for row in choices]);variants=np.repeat(q,2,axis=0);variants[:,2]=choices.ravel();polys=[polygon(row)for row in variants]
   tables=np.ones((15,15,2,2),dtype=bool)
   for i in range(15):
    for j in range(i+1,15):
     if np.linalg.norm(q[i,:2]-q[j,:2])>2:continue
     for x in range(2):
      for y in range(2):tables[i,j,x,y]=polys[2*i+x].intersection(polys[2*j+y]).area<=1e-6
   disks=np.repeat(q[:14],2,axis=0);disks[:,2]=np.tile([0.,np.pi],14)
   best=None;attempts=[]
   for state in range(2):
    piece=q[14].copy();piece[2]=choices[14,state];cx,cy,r=compute_mec_fast(np.vstack([disks,piece]));unary=(radii_at(variants,[cx,cy])<=cap+2e-12).reshape(15,2);unary[14,1-state]=False;assignment=solve_binary_constraints(unary,tables)
    attempts.append(dict(theta14_state=state,enclosure=r,satisfiable=assignment is not None))
    if assignment is None:continue
    candidate=q.copy();candidate[:,2]=choices[np.arange(15),assignment];validation=checked(candidate)
    if validation.valid and validation.score<=cap and(best is None or validation.score<best[0]):best=(validation.score,candidate)
   record=dict(reflection=reflect,quarter=quarter,attempts=attempts,valid=best is not None,seconds=time.monotonic()-start)
   if best is not None:
    filename=out/f'r{int(reflect)}-q{quarter}.json';save(filename,best[1]);record.update(score=best[0],path=str(filename))
   records.append(record);line=json.dumps(record);print(line,flush=True)
   with log.open('a')as f:f.write('\n- '+line+'\n')
 (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
