"""Freeze saved inputs and compare radius separately from exact overlap deficit."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
from breakout_geometry import evaluate_batch
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import AreaConstraints,polygon_areas,save

LEADER=2.9485269238856113
ROOT=Path('experiments/2026-11-04');OUT=ROOT/'round3/rescue-frontier';OUT.mkdir(parents=True,exist_ok=True);SNAP=OUT/'snapshots';SNAP.mkdir(exist_ok=True)
files=[p for root in (ROOT/'round2',ROOT/'round3')for p in root.rglob('*.json')if OUT not in p.parents];started=time.monotonic();records=[];states={};content={};skipped=0
for path in files:
 try:
  raw=path.read_bytes()
  if len(raw)>32768:continue
  data=json.loads(raw)
  if not isinstance(data,list)or len(data)!=15:continue
  if all(isinstance(v,dict)and all(k in v for k in ('x','y','theta'))for v in data):p=np.array([[v['x'],v['y'],v['theta']]for v in data],dtype=float)
  elif all(isinstance(v,list)and len(v)==3 for v in data):p=np.asarray(data,dtype=float)
  else:continue
  if not np.isfinite(p).all():continue
 except (ValueError,OSError,TypeError):skipped+=1;continue
 digest=hashlib.sha256(raw).hexdigest();snapshot=SNAP/(digest+'.json')
 if digest not in content:snapshot.write_bytes(raw);content[digest]=str(snapshot)
 q=np.round(p,6);q[:,2]=[canonicalize_theta(t)for t in q[:,2]];units=np.rint(q*1e6).astype(np.int64);key_units=units.copy();key_units[:,:2]-=key_units[0,:2].copy();key=tuple(key_units.ravel())
 if key in states:records[states[key]]['sources'].append(str(path));continue
 states[key]=len(records);records.append(dict(snapshot=str(snapshot),sha256=digest,sources=[str(path)],raw=p,geometry=q))
print(json.dumps(dict(event='snapshot',files_seen=len(files),pose_inputs=sum(len(r['sources'])for r in records),unique_geometry=len(records),unique_bytes=len(content),seconds=time.monotonic()-started)),flush=True)
(OUT/'sources.json').write_text(json.dumps([{k:v for k,v in r.items()if k not in ('raw','geometry')}for r in records],indent=2)+'\n')
p=np.array([r['geometry']for r in records]);areas,circles,ok=evaluate_batch(p,centers=[0.,0.]);pairs=list(__import__('itertools').combinations(range(15),2));exact_count=0
for i,r in enumerate(records):
 scale=AreaConstraints(p[i]).scale;native=float(np.maximum((areas[i]-1e-6)/scale,0).max());radius=float(circles[i,2]);r.update(radius=radius,radius_above_leader_nm=(radius-LEADER)*1e9,native_maximum_overlap_deficit_nm=native*1e9,native_mec=bool(ok[i]))
 # Exact all pairs for every low-radius point, plus near states used as controls.
 inspect=radius<=LEADER+5e-9 or(native<1e-7 and radius<=LEADER+2e-7)or any('nearest'in source or'new-geometry'in source for source in r['sources'])
 if inspect:
  exact=polygon_areas(p[i]);deficit=np.maximum((exact-1e-6)/scale,0);bad=np.flatnonzero(exact>1e-6);r.update(exact_maximum_overlap_deficit_nm=float(deficit.max()*1e9),exact_overlap_count=len(bad),exact_overlaps=[dict(pair=pairs[k],area=float(exact[k]),deficit_nm=float(deficit[k]*1e9))for k in bad],maximum_estimator_area_error=float(np.abs(exact-areas[i]).max()));exact_count+=1
 r.pop('raw');r.pop('geometry')
 if i%250==0:print(json.dumps(dict(event='progress',completed=i,total=len(records),exact_points=exact_count,seconds=time.monotonic()-started)),flush=True)
exact=[r for r in records if'exact_maximum_overlap_deficit_nm'in r];frontier=[];best=float('inf')
for r in sorted(exact,key=lambda v:(v['radius'],v['exact_maximum_overlap_deficit_nm'])):
 if r['exact_maximum_overlap_deficit_nm']<best-1e-9:frontier.append(r);best=r['exact_maximum_overlap_deficit_nm']
shortlist=[r for r in exact if r['radius']<=LEADER+5e-9 and r['exact_maximum_overlap_deficit_nm']<15]
result=dict(seconds=time.monotonic()-started,leader=LEADER,files_seen=len(files),unique_states=len(records),exact_points=exact_count,native_fallbacks=int((ok==0).sum()),frontier=frontier,shortlist=shortlist,records=records)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('records','frontier','shortlist')}|dict(frontier_count=len(frontier),shortlist_count=len(shortlist))),flush=True)
for r in shortlist:print(json.dumps(dict(event='shortlist',**r)),flush=True)
