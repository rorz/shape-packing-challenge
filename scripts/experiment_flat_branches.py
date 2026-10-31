"""Continuous optima for discrete opposed-flat-angle branches."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.polish import load,AreaConstraints

base=load('experiments/continuous.json');c=AreaConstraints(base)
D=np.zeros((2,46));D[0,5]=1;D[0,17]=-1;D[1,20]=1;D[1,32]=-1
objective=np.zeros(46);objective[-1]=1
for a in [3.141614,3.141615,3.141616,3.141617]:
 for b in [3.141591,3.141592,3.141593,3.141594]:
  target=np.array([a,b]);z=np.r_[base.ravel(),2.948527]
  result=minimize(lambda z:z[-1],z,jac=lambda z:objective,method='SLSQP',constraints=[{'type':'ineq','fun':c.fun,'jac':c.jac},{'type':'eq','fun':lambda z:D@z-target,'jac':lambda z:D}],options={'ftol':2e-13,'maxiter':200})
  p=result.x[:-1].reshape(15,3);path=Path(f'experiments/branch-{a:.6f}-{b:.6f}.json');path.write_text(json.dumps([dict(zip(('x','y','theta'),map(float,s))) for s in p],indent=2)+'\n')
  print(json.dumps({'a':a,'b':b,'radius':result.x[-1],'violation':-min(c.fun(result.x)),'success':result.success,'path':str(path)}),flush=True)
