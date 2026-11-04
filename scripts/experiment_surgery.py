"""Change contact topology by removing two shapes and regrowing replacements."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,ctypes,hashlib,itertools,json,math,subprocess,tempfile,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.polish import load,polish as area_polish,checked,save
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution
LEADER=2.9485269238856113
KERNEL=r'''
#include <cmath>
#include <algorithm>
#include <limits>
struct V{double x,y;V operator+(V b)const{return{x+b.x,y+b.y};}V operator-(V b)const{return{x-b.x,y-b.y};}V operator*(double s)const{return{x*s,y*s};}double dot(V b)const{return x*b.x+y*b.y;}V perp()const{return{-y,x};}};
struct S{double h,hp;};
S support(V u,V n,V t,double r,double cut,bool full){double d=u.dot(n),c=u.dot(t),b=std::sqrt(std::max(0.,1-cut*cut));if(full||d>=cut)return{r,0};return{r*(cut*d+b*std::abs(c)),r*(-cut*c+b*(c>=0?1.:-1.)*d)};}
struct P{double g,j[6];};
P pairgap(V a,V na,V ta,double ra,double ca,bool fa,V b,V nb,V tb,double rb,double cb,bool fb){V d=b-a;P best;best.g=-std::numeric_limits<double>::infinity();
 auto candidate=[&](V u,double pa,double pb,double pax,double pay,double pbx,double pby){S ha=support(u,na,ta,ra,ca,fa),hb=support(u*-1,nb,tb,rb,cb,fb);double g=d.dot(u)-ha.h-hb.h;if(g<=best.g)return;double gp=d.dot(u.perp())-ha.hp-hb.hp;best.g=g;best.j[0]=-u.x+gp*pax;best.j[1]=-u.y+gp*pay;best.j[2]=ha.hp+gp*pa;best.j[3]=u.x+gp*pbx;best.j[4]=u.y+gp*pby;best.j[5]=hb.hp+gp*pb;};
 candidate(na*-1,1,0,0,0,0,0);candidate(nb,0,1,0,0,0,0);
 for(int ia=-1;ia<=1;ia++)for(int ib=-1;ib<=1;ib++){if((fa&&ia)||(fb&&ib))continue;double ba=std::sqrt(std::max(0.,1-ca*ca)),bb=std::sqrt(std::max(0.,1-cb*cb));V ea=ia?(na*ca+ta*(ia*ba))*ra:V{0,0},eb=ib?(nb*cb+tb*(ib*bb))*rb:V{0,0};V q=d-ea+eb;double len=std::hypot(q.x,q.y);if(len<1e-14)continue;V u=q*(1/len),v=u.perp();double pa=ia?((na*(ia*ba)-ta*ca)*ra).dot(v)/len:0,pb=ib?((tb*cb-nb*(ib*bb))*rb).dot(v)/len:0;candidate(u,pa,pb,-v.x/len,-v.y/len,v.x/len,v.y/len);}return best;}
extern "C" void surgery_constraints(int n,const double*z,const double*radii,const double*cuts,const int*full,double*g,double*jac){int cols=3*n+1,rows=n*(n-1)/2+3*n;std::fill(jac,jac+rows*cols,0.);int row=0;
 for(int i=0;i<n;i++){V a={z[3*i],z[3*i+1]},na={std::cos(z[3*i+2]),std::sin(z[3*i+2])},ta=na.perp();for(int j=i+1;j<n;j++){V b={z[3*j],z[3*j+1]},nb={std::cos(z[3*j+2]),std::sin(z[3*j+2])},tb=nb.perp();P p=pairgap(a,na,ta,radii[i],cuts[i],full[i],b,nb,tb,radii[j],cuts[j],full[j]);g[row]=p.g;for(int k=0;k<3;k++){jac[row*cols+3*i+k]=p.j[k];jac[row*cols+3*j+k]=p.j[k+3];}row++;}}
 for(int i=0;i<n;i++){V p={z[3*i],z[3*i+1]},normal={std::cos(z[3*i+2]),std::sin(z[3*i+2])},t=normal.perp();double ri=radii[i],ca=cuts[i],ba=std::sqrt(std::max(0.,1-ca*ca));for(int side=0;side<2;side++){double sign=side?1.:-1.;V far=p+(normal*ca+t*(sign*ba))*ri;double r=std::hypot(far.x,far.y);g[row]=z[3*n]-r;if(r>1e-14){jac[row*cols+3*i]=-far.x/r;jac[row*cols+3*i+1]=-far.y/r;jac[row*cols+3*i+2]=ri*(sign*ba*p.dot(normal)-ca*p.dot(t))/r;}jac[row*cols+3*n]=1;row++;}double r=std::hypot(p.x,p.y);if(full[i]||r<1e-14||p.dot(normal)>=ca*r){g[row]=z[3*n]-r-ri;if(r>1e-14){jac[row*cols+3*i]=-p.x/r;jac[row*cols+3*i+1]=-p.y/r;}jac[row*cols+3*n]=1;}else g[row]=1;row++;}}
'''
class Geometry:
 def __init__(self,radii,full=None,cuts=None):
  self.cuts=np.ascontiguousarray(np.zeros(len(radii))if cuts is None else cuts,dtype=float);self.full=np.ascontiguousarray(np.zeros(len(radii),dtype=np.int32)if full is None else full,dtype=np.int32);self.radii=np.ascontiguousarray(radii,dtype=float);self.n=len(radii);self.rows=self.n*(self.n-1)//2+3*self.n;self.cols=3*self.n+1;self.last=None
  digest=hashlib.sha256(KERNEL.encode()).hexdigest()[:16];path=Path(tempfile.gettempdir())/f'packing-surgery-{digest}.so'
  if not path.exists():
   source=path.with_suffix('.cpp');source.write_text(KERNEL);temporary=path.with_suffix(f'.{os.getpid()}.tmp.so');subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,capture_output=True);temporary.replace(path)
  self.lib=ctypes.CDLL(str(path));self.fn=self.lib.surgery_constraints;array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');self.fn.argtypes=[ctypes.c_int,array,array,array,integer,array,array]
 def evaluate(self,z):
  z=np.ascontiguousarray(z,dtype=float)
  if self.last is not None and np.array_equal(z,self.last):return self.g,self.j
  self.g=np.empty(self.rows);self.j=np.empty((self.rows,self.cols));self.fn(self.n,z,self.radii,self.cuts,self.full,self.g,self.j);self.last=z.copy();return self.g,self.j
 def fun(self,z):return self.evaluate(z)[0]
 def jac(self,z):return self.evaluate(z)[1]

def write_full(path,p):Path(path).write_text(json.dumps([dict(zip(('x','y','theta'),map(float,row)))for row in p],indent=2)+'\n')
def optimize(g,p,R,iterations=120):
 z=np.r_[p.ravel(),R];direction=np.zeros(len(z));direction[-1]=1
 sol=minimize(lambda z:z[-1],z,jac=lambda z:direction,method='SLSQP',constraints=dict(type='ineq',fun=g.fun,jac=g.jac),bounds=[(None,None)]*(len(z)-1)+[(1,4)],options=dict(ftol=2e-11,maxiter=iterations));return sol.x[:-1].reshape(-1,3),float(sol.x[-1]),float(g.fun(sol.x).min()),int(sol.nit)
def relax(g,p,R,iterations=180):
 def objective(v):
  gaps,jac=g.evaluate(np.r_[v,R]);neg=np.minimum(gaps,0);return .5*float(neg@neg),neg@jac[:,:-1]
 sol=minimize(objective,p.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=iterations,ftol=1e-14,gtol=2e-7,maxls=30));return sol.x.reshape(-1,3),float(sol.fun),int(sol.nit)
def fingerprint(g,p,R):
 gaps,_=g.evaluate(np.r_[p.ravel(),R]);n=len(p);pairs=list(itertools.combinations(range(n),2));neighbors=[[]for _ in p]
 for k,(i,j)in enumerate(pairs):
  if gaps[k]<3e-6:neighbors[i].append(j);neighbors[j].append(i)
 boundary=gaps[len(pairs):].reshape(n,3);labels=[repr((len(neighbors[i]),int((boundary[i]<3e-6).sum())))for i in range(n)]
 for _ in range(4):
  descriptors=[(labels[i],tuple(sorted(labels[j]for j in neighbors[i])))for i in range(n)];labels=[hashlib.sha256(repr(v).encode()).hexdigest()[:24]for v in descriptors]
 return hashlib.sha256(repr((sorted(labels),sorted(len(v)for v in neighbors))).encode()).hexdigest()[:16]
def pocket(existing,R,rng,samples):
 n=len(existing)+1;g=Geometry(np.ones(n));best=[];angles=rng.uniform(-np.pi,np.pi,samples);rad=R*np.sqrt(rng.uniform(0,1,samples));az=rng.uniform(-np.pi,np.pi,samples);candidates=np.c_[rad*np.cos(az),rad*np.sin(az),angles]
 # Only new-shape pair rows and boundary rows matter; frozen mutual gaps are irrelevant.
 pairrows=[k for k,(i,j)in enumerate(itertools.combinations(range(n),2))if j==n-1];rows=np.r_[pairrows,np.arange(g.rows-3,g.rows)]
 for candidate in candidates:
  p=np.vstack([existing,candidate]);gaps,_=g.evaluate(np.r_[p.ravel(),R]);best.append(float(gaps[rows].min()))
 order=np.argsort(best);chosen=int(rng.choice(order[-min(8,len(order)):]))
 return candidates[chosen],best[chosen]
def audit(out):
 rng=np.random.default_rng(81104);errors=[];jacerrors=[]
 for n in [3,13,15]:
  native=FastGeometry(n);g=Geometry(np.ones(n))
  for _ in range(20):
   p=rng.normal(size=(n,3));z=np.r_[p.ravel(),3.1];a,aj=g.evaluate(z);b,bj=native.evaluate(z);bb,bjb=native.boundary(z);pairs=n*(n-1)//2;errors.append(float(np.max(np.abs(a-np.r_[b[:pairs],bb]))));jacerrors.append(float(np.max(np.abs(aj-np.vstack([bj[:pairs],bjb])))))
 # Unequal-radius finite differences test shape-angle and position derivatives.
 g=Geometry([.2,.6,1]);p=np.array([[-1.2,.1,.4],[.2,-.4,-.3],[1.,.5,2.1]]);z=np.r_[p.ravel(),3.2];v,j=g.evaluate(z);unequal=[];h=1e-7
 for k in range(len(z)):
  d=np.zeros(len(z));d[k]=h;fd=(g.fun(z+d)-g.fun(z-d))/(2*h);unequal.append(float(np.max(np.abs(fd-j[:,k]))))
 result=dict(seed=81104,unit_value_error=max(errors),unit_jacobian_error=max(jacerrors),unequal_radius_derivative_error=max(unequal));(out/'kernel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='audit',**result)),flush=True)
 if max(errors)>1e-11 or max(jacerrors)>1e-8 or max(unequal)>1e-6:raise ValueError('Variable-radius kernel audit failed')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=8110401);ap.add_argument('--input',default='solution.json');ap.add_argument('--output',default='experiments/2026-11-04/round2/topology');ap.add_argument('--container',type=float,default=3.12);ap.add_argument('--pockets',type=int,default=512);ap.add_argument('--audit-only',action='store_true');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);audit(out)
 if a.audit_only:return
 rng=np.random.default_rng(a.seed);started=time.monotonic();origin=load(a.input);write_full(out/'input.json',origin);unit=Geometry(np.ones(15));origin,baseline,gap,nit=optimize(unit,origin,3.0);write_full(out/'strict-baseline.json',origin);signature=fingerprint(unit,origin,baseline);archive={signature:(origin.copy(),baseline)};best=baseline;best_area=float('inf');trials=0;order=list(itertools.combinations(range(15),2));rng.shuffle(order);order=[(1,5),(6,10)]+[p for p in order if p not in [(1,5),(6,10)]]
 print(json.dumps(dict(event='baseline',radius=baseline,gap=gap,iterations=nit,fingerprint=signature)),flush=True)
 with (out/'trials.jsonl').open('w')as recordfile:
  while time.monotonic()-started<a.seconds:
   pair=order[trials%len(order)];parent_signature=signature if rng.random()<.65 else rng.choice(list(archive));parent,parentR=archive[parent_signature];keep=[i for i in range(15)if i not in pair];skeleton,smallR,smallgap,smallnit=optimize(Geometry(np.ones(13)),parent[keep],parentR,iterations=100);proposals=[];scores=[];existing=skeleton.copy()
   for _ in range(2):
    q,s=pocket(existing,a.container,rng,a.pockets);proposals.append(q);scores.append(s);existing=np.vstack([existing,q])
   p=existing;stages=[]
   for size in [.15,.35,.55,.75,1.]:
    radii=np.r_[np.ones(13),size,size];geometry=Geometry(radii);p,loss,it=relax(geometry,p,a.container,iterations=180);stages.append(dict(size=size,loss=loss,iterations=it))
    if loss>.03:break
   p,R,mingap,nit=optimize(unit,p,a.container,iterations=160);finger=fingerprint(unit,p,R);novel=finger not in archive;feasible=mingap>-2e-7
   row=dict(trial=trials,seed=a.seed,removed=list(pair),parent=parent_signature,skeleton_radius=smallR,skeleton_gap=smallgap,pocket_scores=scores,stages=stages,radius=R,gap=mingap,iterations=nit,fingerprint=finger,novel=novel,feasible=feasible,seconds=time.monotonic()-started)
   if feasible and R<3.15:
    if novel or R<archive[finger][1]:archive[finger]=(p.copy(),R);write_full(out/f'topology-{finger}.json',p)
    if R<best:best=R;write_full(out/'strict-best.json',p);row['strict_best']=True
    if R<baseline+2e-6:
     polished=area_polish(p,maxiter=180);q=polished.x[:-1].reshape(15,3);pr=float(polished.x[-1]);violation=float(max(0,-min(AreaConstraints_cached(q,pr))));row['area_radius']=pr;row['area_violation']=violation
     if violation<2e-8 and pr<best_area:best_area=pr;write_full(out/'continuous-best.json',q)
     v=checked(q);row['rounded_valid']=v.valid;row['rounded_score']=v.score
     if v.valid and v.score<2.948526966725367:
      save(out/'valid-best.json',q);response=check_solution(json.loads((out/'valid-best.json').read_text()));(out/'api.json').write_text(json.dumps(response,indent=2)+'\n');row['api']=response
      if response.get('valid')and response.get('score',99)<LEADER:recordfile.write(json.dumps(row)+'\n');recordfile.flush();print(json.dumps(row),flush=True);break
   trials+=1;recordfile.write(json.dumps(row)+'\n');recordfile.flush();print(json.dumps(row),flush=True)
 print(json.dumps(dict(event='done',trials=trials,topologies=len(archive),strict_best=best,continuous_best=best_area,seconds=time.monotonic()-started)),flush=True)
def AreaConstraints_cached(p,R):
 from semicircle_packing.polish import AreaConstraints
 return AreaConstraints(p).fun(np.r_[p.ravel(),R])
if __name__=='__main__':main()
