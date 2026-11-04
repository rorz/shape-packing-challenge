"""Conservative pair relaxation with exact GEOS completion at selected cells."""
from __future__ import annotations
import itertools,time
import numpy as np
import shapely
from conflict_area_band import true_area_loss_band
from semicircle_packing.geometry import Semicircle,semicircle_polygon


class PairOracle:
 def __init__(self,domains,*,guard=1e-9,deadline=None):
  if guard<1e-9:raise ValueError('Arithmetic guard cannot weaken the frozen band contract')
  self.domains=domains;self.status={};self.polygons={};self.exact_areas={};self.stats=dict(native_pairs=0,initial_uncertain=0,certain_allowed=0,certain_rejected=0,geos_pairs=0,exact_allowed=0,exact_rejected=0)
  for i,j in itertools.combinations(range(len(domains)),2):
   if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Conservative native band preparation reached deadline')
   a,b=domains[i],domains[j];gap=np.maximum(np.maximum(a[:,:2].min(0)-b[:,:2].max(0),b[:,:2].min(0)-a[:,:2].max(0)),0)
   if np.linalg.norm(gap)>2:continue
   area,loss=true_area_loss_band(a,b);allowed=area<=1e-6-guard;rejected=area>1e-6+loss+guard;state=np.zeros(area.shape,dtype=np.int8);state[allowed]=1;state[rejected]=-1;self.status[i,j]=state;self.stats['native_pairs']+=state.size;self.stats['initial_uncertain']+=int((state==0).sum());self.stats['certain_allowed']+=int(allowed.sum());self.stats['certain_rejected']+=int(rejected.sum())
 def relaxed_tables(self):
  return[(i,j,np.column_stack(np.nonzero(state>=0)))for(i,j),state in self.status.items()if not(state>=0).all()]
 def polygon(self,piece,value):
  key=piece,value
  if key not in self.polygons:self.polygons[key]=semicircle_polygon(Semicircle(*self.domains[piece][value]))
  return self.polygons[key]
 def check(self,assignment):
  """Return exact/proven violations; uncertain cells alone invoke GEOS."""
  selected=[];violations=[]
  for(i,j),state in self.status.items():
   a,b=int(assignment[i]),int(assignment[j]);value=state[a,b]
   if value==0:selected.append((i,j,a,b))
   elif value<0:violations.append(dict(pair=[i,j],values=[a,b],area=self.exact_areas.get((i,j,a,b)),cached=True))
  if selected:
   pa=np.asarray([self.polygon(i,a)for i,j,a,b in selected],dtype=object);pb=np.asarray([self.polygon(j,b)for i,j,a,b in selected],dtype=object);areas=shapely.area(shapely.intersection(pa,pb));self.stats['geos_pairs']+=len(selected)
   for key,area in zip(selected,areas):
    i,j,a,b=key;area=float(area);good=area<=1e-6;self.status[i,j][a,b]=1 if good else-1;self.exact_areas[key]=area;self.stats['exact_allowed'if good else'exact_rejected']+=1
    if not good:violations.append(dict(pair=[i,j],values=[a,b],area=area,cached=False))
  return violations
