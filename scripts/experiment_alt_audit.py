"""Reproduce the analytic flat-clearance derivative audit and runtime manifest."""
import hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np
from experiment_alt_quadratic import flat_clearance
from semicircle_packing.polish import load

out=Path('experiments/2026-11-04/alternative');rng=np.random.default_rng(110406);input_path=out/'quadratic-current/candidate-0.json';p=load(input_path);errors=[];h=1e-7
for i,j in [(1,5),(6,10)]:
 for _ in range(40):
  q=p.copy();q[[i,j],:2]+=rng.uniform(-50e-6,50e-6,(2,2));q[[i,j],2]+=rng.uniform(-50e-6,50e-6);_,derivative=flat_clearance(q,i,j)
  for direction in [(i,0),(i,1),(j,0),(j,1),('common',2)]:
   a=q.copy();b=q.copy()
   if direction[0]=='common':a[[i,j],2]+=h;b[[i,j],2]-=h;pred=derivative[3*i+2]
   else:k,d=direction;a[k,d]+=h;b[k,d]-=h;pred=derivative[3*k+d]
   fd=(flat_clearance(a,i,j)[0]-flat_clearance(b,i,j)[0])/(2*h);errors.append(abs(fd-pred))
result=dict(seed=110406,input=str(input_path),placements=80,derivative_directions=len(errors),finite_difference_step=h,max_absolute_derivative_error=max(errors),median_absolute_derivative_error=float(np.median(errors)))
(out/'flat-derivative-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
versions={name:importlib.metadata.version(name)for name in ['numpy','scipy','shapely','fpylll','pyscipopt']};scripts={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path('scripts').glob('experiment_alt_*.py')};inputs={str(input_path):hashlib.sha256(input_path.read_bytes()).hexdigest()};(out/'environment.json').write_text(json.dumps(dict(versions=versions,scripts=scripts,inputs=inputs),indent=2)+'\n')
