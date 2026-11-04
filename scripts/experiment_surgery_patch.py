"""Find rigidly matching subpatches between equal-radius packings."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from semicircle_packing.polish import load
ROOT=Path('experiments/2026-11-04/round2/topology')
def transformed(p,phi,translation,reflected):
 q=p.copy()
 if reflected:q[:,1]*=-1;q[:,2]*=-1
 co,si=np.cos(phi),np.sin(phi);rot=np.array([[co,si],[-si,co]]);q[:,:2]=q[:,:2]@rot+translation;q[:,2]+=phi;return q

def consensus(p,q,epsilon=1e-3):
 best=None
 for reflected in [False,True]:
  source=p.copy()
  if reflected:source[:,1]*=-1;source[:,2]*=-1
  for i,j in itertools.product(range(len(p)),range(len(q))):
   phi=q[j,2]-source[i,2];co,si=np.cos(phi),np.sin(phi);rot=np.array([[co,si],[-si,co]]);translation=q[j,:2]-source[i,:2]@rot;X=source[:,:2]@rot+translation;d=np.linalg.norm(X[:,None,:]-q[None,:,:2],axis=2);angles=np.abs(np.arctan2(np.sin(source[:,None,2]+phi-q[None,:,2]),np.cos(source[:,None,2]+phi-q[None,:,2])));match=(d<epsilon)&(angles<epsilon);cost=-1000*match+np.minimum(d*d+angles*angles,10);rows,cols=linear_sum_assignment(cost);inliers=match[rows,cols];count=int(inliers.sum());error=float(np.sum((d[rows[inliers],cols[inliers]]**2+angles[rows[inliers],cols[inliers]]**2)))
   if best is None or(count,-error)>(best['count'],-best['error']):best=dict(count=count,error=error,reflected=reflected,phi=float(np.arctan2(np.sin(phi),np.cos(phi))),translation=translation.tolist(),mapping=cols.tolist(),inliers=inliers.tolist(),center_errors=d[rows,cols].tolist(),angle_errors=angles[rows,cols].tolist())
 return best

def main():
 p=load(ROOT/'continuous-best.json');q=np.load('experiments/soft_modes.npz')['p'];outer=consensus(p,q);aligned=transformed(p,outer['phi'],np.array(outer['translation']),outer['reflected']);source_patch=np.flatnonzero(~np.array(outer['inliers']));target_patch=np.array(outer['mapping'])[source_patch];patch=consensus(aligned[source_patch],q[target_patch]);co,si=np.cos(patch['phi']),np.sin(patch['phi']);A=np.array([[co,-si],[si,co]])@np.diag([1,-1]if patch['reflected']else[1,1]);t=np.array(patch['translation']);reflection=None
 if patch['reflected']:
  eigenvectors=np.linalg.eigh(A)[1];axis=eigenvectors[:,1];normal=eigenvectors[:,0];reflection=dict(axis_direction=axis.tolist(),normal=normal.tolist(),line_offset=float(t@normal/2),glide_along_axis=float(t@axis))
 result=dict(outer=outer,source_patch=source_patch.tolist(),target_patch=target_patch.tolist(),patch=patch,patch_reflection=reflection,epsilon=1e-3);(ROOT/'patch-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
