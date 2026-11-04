"""Measure rigid-motion curvature and the contact stress budget."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints,load
from semicircle_packing.fast_mec import compute_mec_fast

OUT=Path('experiments/2026-11-04/alternative/geometry.json')
def native(p,c,R):
 cx,cy,r=compute_mec_fast(p);centered=p.copy();centered[:,:2]-=[cx,cy];g,j=c.evaluate(np.r_[centered.ravel(),R]);return g,j,r

def main():
 state=np.load('experiments/soft_modes.npz');p=state['p'].copy();cx,cy,R=compute_mec_fast(p);p[:,:2]-=[cx,cy];c=AreaConstraints(p);g,j,_=native(p,c,R);active=g<1e-7;J=j[active,:45];U,S,V=np.linalg.svd(J,full_matrices=True)
 # Infinitesimal Euclidean motions, with all orientation components advancing under rotation.
 E=np.zeros((45,3));E[0::3,0]=1;E[1::3,1]=1;E[0::3,2]=-p[:,1];E[1::3,2]=p[:,0];E[2::3,2]=1
 left=U[:,-1];left*=np.sign(left.sum());radius_column=j[active,-1];stress_budget=(2.9485269238856113-R)*(left@radius_column)
 singular=dict(radius=R,contacts=int(active.sum()),singular_values=S.tolist(),rigid_derivative_norms=np.linalg.norm(J@E,axis=0).tolist(),stress_min=float(left.min()),stress_sum=float(left.sum()),weighted_leader_slack=float(stress_budget),leader_headroom=float(2.9485269238856113-R))
 modes=[]
 for index in range(39,45):
  mode=V[index];coeff=np.linalg.lstsq(E,mode,rcond=None)[0];residual=mode-E@coeff
  modes.append(dict(index=index,singular_value=float(S[index])if index<len(S)else 0,rigid_fraction=float(1-np.linalg.norm(residual)**2),first_flat=float(mode[5]-mode[17]),second_flat=float(mode[20]-mode[32])))
 rows=[]
 for path in ['experiments/lll_hillclimb_best_state.npz','experiments/lll_hillclimb_projected_state.npz']:
  st=np.load(path);anchor=st['placements'].copy();cx,cy,r=compute_mec_fast(anchor);anchor[:,:2]-=[cx,cy];ac=AreaConstraints(st['origin']);ga,ja,ra=native(anchor,ac,2.9485269238856113)
  EE=E.copy();EE[0::3,2]=-anchor[:,1];EE[1::3,2]=anchor[:,0]
  for index in np.argsort(np.linalg.norm(st['moves'].reshape(45,45),axis=1))[-8:]:
   delta=st['moves'][index]*1e-6;coef=np.linalg.lstsq(EE,delta.ravel(),rcond=None)[0];rigid=EE@coef;residual=delta.ravel()-rigid;alpha=coef[2]
   trial=anchor+delta;actual,jactual,rad=native(trial,ac,2.9485269238856113);pred=ga[:105]+ja[:105,:45]@delta.ravel();rot=anchor.copy();cs,sn=np.cos(alpha),np.sin(alpha);rot[:,:2]=rot[:,:2]@np.array([[cs,sn],[-sn,cs]]);rot[:,2]+=alpha
   linearrot=anchor+(EE[:,2]*alpha).reshape(15,3);curvature=rot-linearrot
   rows.append(dict(checkpoint=path,index=int(index),move_norm_micro=float(np.linalg.norm(delta)*1e6),alpha_micro=float(alpha*1e6),rigid_fraction=float(np.linalg.norm(rigid)**2/np.linalg.norm(delta)**2),rotation_curvature_max_nm=float(np.abs(curvature[:,:2]).max()*1e9),actual_area_violation_nm=float(max(0,-actual[:105].min())*1e9),linear_area_violation_nm=float(max(0,-pred.min())*1e9),area_linearization_error_nm=float(np.max(np.abs(actual[:105]-pred))*1e9),radius=float(rad)))
 result=dict(contact_model=singular,soft_modes=modes,large_moves=rows);OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
