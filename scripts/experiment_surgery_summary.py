"""Summarize bounded surgery runs and compare their competitive geometries."""
import hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from semicircle_packing.polish import load

ROOT=Path('experiments/2026-11-04/round2/topology')
def alignment(source,target):
 best=None
 for reflected in [False,True]:
  p=source.copy()
  if reflected:p[:,1]*=-1;p[:,2]*=-1
  C=p[:,:2]+4/(3*np.pi)*np.c_[np.cos(p[:,2]),np.sin(p[:,2])];N=np.c_[np.cos(p[:,2]),np.sin(p[:,2])];D=target[:,:2]+4/(3*np.pi)*np.c_[np.cos(target[:,2]),np.sin(target[:,2])];M=np.c_[np.cos(target[:,2]),np.sin(target[:,2])]
  for initial in np.r_[np.linspace(-np.pi,np.pi,32,endpoint=False),(target[:,2,None]-p[None,:,2]).ravel()]:
   phi=initial
   for _ in range(20):
    cs,sn=np.cos(phi),np.sin(phi);rotation=np.array([[cs,sn],[-sn,cs]]);X=C@rotation;shift=D.mean(axis=0)-X.mean(axis=0);NN=N@rotation;cost=np.sum((X[:,None,:]+shift-D[None,:,:])**2,axis=2)+.2*np.sum((NN[:,None,:]-M[None,:,:])**2,axis=2);_,order=linear_sum_assignment(cost);x=C-C.mean(axis=0);y=D[order]-D[order].mean(axis=0);cross=np.sum((y[:,0]+1j*y[:,1])*np.conj(x[:,0]+1j*x[:,1]))+.2*np.sum((M[order,0]+1j*M[order,1])*np.conj(N[:,0]+1j*N[:,1]));newphi=np.angle(cross)
    if abs(np.arctan2(np.sin(newphi-phi),np.cos(newphi-phi)))<1e-13:phi=newphi;break
    phi=newphi
   cs,sn=np.cos(phi),np.sin(phi);rotation=np.array([[cs,sn],[-sn,cs]]);shift=D[order].mean(axis=0)-C.mean(axis=0)@rotation;position=p[:,:2]@rotation+shift-target[order,:2];angles=np.arctan2(np.sin(p[:,2]+phi-target[order,2]),np.cos(p[:,2]+phi-target[order,2]));error=float(np.sqrt(np.mean(np.sum(position**2,axis=1)+angles**2)))
   if best is None or error<best['rms_configuration_error']:best=dict(reflected=reflected,rotation=phi,rms_configuration_error=error,max_center_error=float(np.max(np.linalg.norm(position,axis=1))),max_angle_error=float(np.max(np.abs(angles))))
 return best

def summarize(path,kind):
 rows=[json.loads(line)for line in path.read_text().splitlines()];trials=[r for r in rows if r.get('event')!='fusion'];feasible=[r for r in trials if r.get('feasible',r['gap']>-2e-7)];other=[r for r in feasible if r['radius']>2.948682774628151];areas=[r for r in trials if 'area_radius'in r and r.get('area_violation',1)<2e-8]
 return dict(kind=kind,trials=len(trials),strict_feasible=len(feasible),strict_best=min((r['radius']for r in feasible),default=None),best_other_strict=min((r['radius']for r in other),default=None),continuous_best=min((r['area_radius']for r in areas),default=None),rounded_valid=sum(r.get('rounded_valid',False)for r in trials),seconds=rows[-1]['seconds'],contact_signatures=len({r['fingerprint']for r in feasible}))
def main():
 runs=[]
 for path,kind in [(ROOT/'trials.jsonl','remove_regrow'),(ROOT/'split/trials.jsonl','disk_fission'),(ROOT/'caps/trials.jsonl','area_transfer')]:
  if path.exists():runs.append(summarize(path,kind))
 target=np.load('experiments/soft_modes.npz')['p'];alignments={}
 for path in [ROOT/'continuous-best.json',ROOT/'split/continuous-best.json',ROOT/'caps/continuous-best.json']:
  if path.exists():alignments[str(path)]=alignment(load(path),target)
 graph_audit=json.loads((ROOT/'graph-isomorphism-audit.json').read_text());graph_audit.pop('classes');report=dict(valid_improvement=False,runs=runs,total_trials=sum(r['trials']for r in runs),graph_equivalence_audit=graph_audit,competitive_geometry_alignment=alignments,alignment_reference='experiments/soft_modes.npz',negative_inference='No better basin found within these three bounded surgery families; this does not prove a global optimum or that the leader uses this basin.',untested_extension='Optimize11 semicircles plus2 full disks; enumerate joint contact partitions of both disks, then release all4 halves together. This coupled four-piece move was not tested.');report['negative_inference']='No lower radius found. Removal/regrowth produced a noncongruent equal-radius topology via a three-piece patch reflection. No proof of a global optimum or of the leader using this basin.'
 if (ROOT/'inventory/summary.json').exists():
  inventory=json.loads((ROOT/'inventory/summary.json').read_text());inventory.pop('records');report['inventory']=inventory
 if (ROOT/'noncongruent-seed-audit.json').exists():report['noncongruent_seed']=json.loads((ROOT/'noncongruent-seed-audit.json').read_text())
 (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 versions={name:importlib.metadata.version(name)for name in ['numpy','scipy','shapely','networkx']};scripts={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path('scripts').glob('experiment_surgery*.py')};inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/'input.json',Path('experiments/soft_modes.npz')]};(ROOT/'environment.json').write_text(json.dumps(dict(versions=versions,scripts=scripts,inputs=inputs),indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
