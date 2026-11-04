"""Verify elite provenance and search one new feasible center state."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,shlex,subprocess,sys
from pathlib import Path
import numpy as np
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import checked,load,save
from arena import check_solution


def center_signature(p,reflections=False):
 """Unlabelled integer centers, modulo translation and square symmetries."""
 xy=np.rint(p[:,:2]*1e6).astype(np.int64);keys=[]
 for reflected in range(2 if reflections else 1):
  q=xy.copy()
  if reflected:q[:,0]*=-1
  for _ in range(4):
   q=q[:,[1,0]]*[-1,1];r=q-q.min(axis=0)
   keys.append(tuple(r[np.lexsort((r[:,1],r[:,0]))].ravel().tolist()))
 return min(keys)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--exclude',action='append',default=['solution.json']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=3110461);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 excluded=[(name,load(name))for name in a.exclude];signatures={center_signature(p,True)for _,p in excluded};manifest=json.loads(a.manifest.read_text());audit=[];selected=None
 for item in sorted(manifest['entries'],key=lambda e:e['score']):
  path=a.manifest.parent/item['file'];p=load(str(path));validation=checked(p)
  with np.load(a.manifest.parent/item['sidecar'])as state:
   delta=(state['coefficients']@state['moves'].reshape(-1,45)).reshape(15,3);delta[:,:2]-=delta[0,:2].copy();recovered=state['anchor_units'].reshape(15,3)+delta;recovered[:,2]=np.rint([canonicalize_theta(t*1e-6)*1e6 for t in recovered[:,2]]).astype(np.int64)
   provenance_ok=np.array_equal(recovered.ravel(),state['units'].ravel());moves=state['moves'].reshape(-1,15,3).copy()
  assert provenance_ok and validation.valid and abs(validation.score-item['score'])<2e-12
  matches=[name for name,q in excluded if center_signature(p,True)==center_signature(q,True)]
  audit.append(dict(file=str(path),score=validation.score,reference_valid=True,provenance_ok=provenance_ok,square_symmetry_center_matches=matches))
  if selected is None and center_signature(p,True)not in signatures:selected=(item,p,moves,path)
 (a.output/'distinctness.json').write_text(json.dumps(dict(excluded=a.exclude,states=audit),indent=2)+'\n')
 if selected is None:
  print(json.dumps(dict(event='no_distinct_elite',states=len(audit))),flush=True);return
 item,p,moves,path=selected;save(a.output/'start.json',p);np.savez_compressed(a.output/'checkpoint.npz',moves=moves)
 command=[sys.executable,str(Path(__file__).with_name('experiment_breakout.py')),'--input',str(a.output/'start.json'),'--checkpoint',str(a.output/'checkpoint.npz'),'--output',str(a.output/'best.json'),'--seconds',str(a.seconds),'--seed',str(a.seed),'--target','2.9485269238','--movement','1000','--combinations','1000']
 (a.output/'steps.md').write_text('# Elite relay follow-up\n\nThe selected start is reference-valid, reconstructs exactly from its saved frame and decoded lattice coefficients, and differs from each excluded incumbent under unlabelled square-lattice center symmetries and integer translation.\n\nSource: '+str(path)+'\n\n```text\n'+shlex.join(command)+'\n```\n')
 print(json.dumps(dict(event='selected',source=str(path),score=item['score'],command=command)),flush=True)
 with (a.output/'breakout.log').open('w')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
 summary=dict(event='done',exit_code=result.returncode,source=str(path),source_score=item['score'],distinct_xy=True)
 if (a.output/'best.json').exists():
  validation=checked(load(str(a.output/'best.json')));summary.update(reference_valid=validation.valid,reference_score=validation.score)
  if validation.valid and validation.score<checked(load('solution.json')).score:
   response=check_solution(json.loads((a.output/'best.json').read_text()));(a.output/'best-api.json').write_text(json.dumps(response,indent=2)+'\n');summary['api']=response
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
