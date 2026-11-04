"""GEOS-verified radius/overlap archive; distant checks are coalesced in time."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import polygon_areas,save
from semicircle_packing.fast_mec import compute_mec_fast

LEADER=2.9485269238856113
PAIRS=[(i,j)for i in range(15)for j in range(i+1,15)]


class OverlapFrontier:
 def __init__(self,path,width_nm=1.,archive_guard_nm=100.):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
  self.width=width_nm;self.guard=archive_guard_nm;self.low=-60.;self.high=45.
  self.bins={};self.seen=set();self.pending={};self.deferred_valid=[];self.serial=0
  self.counts=dict(in_radius_window=0,native_under20nm=0,geos_checked=0,geos_under20nm=0,saved=0,distant_nominations=0,distant_superseded=0,seconds_geos=0.)
  self.last_write=0.;self.last_drain=time.monotonic()

 def bin(self,radius):return int(np.floor(((radius-LEADER)*1e9-self.low)/self.width))

 def snapshot(self,index,actual,areas,circles,scale,maximum,bad,raw_at,state_at):
  # Capture all frame-dependent values now: the frame is replaced next batch.
  state={key:value.copy()if isinstance(value,np.ndarray)else value for key,value in state_at(index).items()}
  return dict(geometry=actual[index].copy(),areas=areas[index].copy(),circle=circles[index].copy(),scale=scale.copy(),native=float(maximum[index]),native_bad=int(bad[index]),raw=raw_at(index).copy(),state=state)

 def consider(self,actual,areas,circles,scale,raw_at,state_at):
  low=LEADER+self.low*1e-9;high=LEADER+self.high*1e-9
  indices=np.flatnonzero((circles[:,2]>=low)&(circles[:,2]<=high));valid=[]
  if not len(indices):
   if time.monotonic()-self.last_drain>10:self.flush()
   return valid
  native=np.maximum(areas-1e-6,0)/scale[None,:]*1e9;maximum=native.max(axis=1)
  bad=(areas>1e-6+1e-11).sum(axis=1);bins=np.floor(((circles[:,2]-LEADER)*1e9-self.low)/self.width).astype(int)
  self.counts['in_radius_window']+=len(indices);self.counts['native_under20nm']+=int((maximum[indices]<20).sum())
  indices=indices[maximum[indices]<self.guard];proposed=set()
  for b in np.unique(bins[indices]):
   choices=indices[bins[indices]==b]
   proposed.add(int(choices[np.argmin(maximum[choices])]))
   order=np.lexsort((maximum[choices],bad[choices]));proposed.add(int(choices[order[0]]))
  proposed.update(map(int,indices[maximum[indices]<25]))
  for index in sorted(proposed,key=lambda k:(maximum[k],circles[k,2])):
   b=int(bins[index]);entry=self.bins.get(b,{})
   old=entry.get('overlap');old_count=entry.get('count')
   old_max=float('inf')if old is None else old['max_overlap_nm'];old_bad=16 if old_count is None else old_count['violated_pairs']
   near=maximum[index]<25 and maximum[index]<old_max+10
   improving=maximum[index]<old_max-1;fewer=bad[index]<old_bad
   if not(near or improving or fewer):continue
   if maximum[index]<25:
    point=self.snapshot(index,actual,areas,circles,scale,maximum,bad,raw_at,state_at)
    if self.verify(point):valid.append(index)
    continue
   # Keep the best distant contenders in memory until a ten-second flush.
   # This coalesces repeated bin improvements without losing their source basis.
   ranks={'overlap':(maximum[index],circles[index,2]),'count':(bad[index],maximum[index],circles[index,2])}
   selected=[]
   for kind,rank in ranks.items():
    old=self.pending.get((b,kind))
    if old is None or rank<old[0]:selected.append((kind,rank))
   if selected:
    point=self.snapshot(index,actual,areas,circles,scale,maximum,bad,raw_at,state_at)
    for kind,rank in selected:
     self.counts['distant_superseded']+=int((b,kind)in self.pending)
     self.pending[b,kind]=(rank,point);self.counts['distant_nominations']+=1
  if time.monotonic()-self.last_drain>10:self.flush()
  return valid

 def verify(self,point):
  actual=point['geometry'];scale=point['scale'];circle=point['circle']
  units=np.rint(actual*1e6).astype(np.int64);units[:,:2]-=units[0,:2]
  key=hashlib.sha256(units.tobytes()).digest()
  if key in self.seen:return False
  self.seen.add(key);started=time.perf_counter();geos=polygon_areas(actual)
  cx,cy,R=compute_mec_fast(actual,initial_center=circle[:2])
  self.counts['seconds_geos']+=time.perf_counter()-started;self.counts['geos_checked']+=1
  excess=np.maximum(geos-1e-6,0)/scale*1e9;overlap=float(excess.max());violated=np.flatnonzero(geos>1e-6)
  self.counts['geos_under20nm']+=int(overlap<20);valid=not len(violated)
  if not self.low<=(R-LEADER)*1e9<=self.high:return valid
  b=self.bin(R);entry=self.bins.setdefault(b,{})
  old=entry.get('overlap');old_count=entry.get('count')
  wins_overlap=old is None or(overlap,R)<(old['max_overlap_nm'],old['radius'])
  wins_count=old_count is None or(len(violated),overlap,R)<(old_count['violated_pairs'],old_count['max_overlap_nm'],old_count['radius'])
  if not(wins_overlap or wins_count):return valid
  name=f'witness-{self.serial:05d}';self.serial+=1;save(self.path/(name+'.json'),point['raw'])
  context=point['state'];context.update(placements=point['raw'],geometry=actual,geos_areas=geos,area_scale=scale,circle=np.array([cx,cy,R]))
  np.savez_compressed(self.path/(name+'.npz'),**context)
  record=dict(file=name+'.json',sidecar=name+'.npz',radius=R,radius_delta_nm=(R-LEADER)*1e9,max_overlap_nm=overlap,sum_overlap_nm2=float(excess@excess),violated_pairs=len(violated),pairs=[list(PAIRS[k])for k in violated],max_area=float(geos.max()),native_max_overlap_nm=point['native'],raw_sha256=hashlib.sha256((self.path/(name+'.json')).read_bytes()).hexdigest())
  if wins_overlap:entry['overlap']=record
  if wins_count:entry['count']=record
  self.counts['saved']+=1
  if time.monotonic()-self.last_write>1:self.write_manifest()
  return valid

 def flush(self):
  pending=self.pending;self.pending={}
  for _,point in sorted(pending.values(),key=lambda item:item[1]['native']):
   if self.verify(point):self.deferred_valid.append(point)
  self.last_drain=time.monotonic();self.write_manifest()

 def write_manifest(self):
  records={r['file']:r for v in self.bins.values()for r in v.values()}
  rows=sorted(records.values(),key=lambda r:r['radius']);pareto=[];best=float('inf')
  for row in rows:
   if row['max_overlap_nm']<best:pareto.append(row);best=row['max_overlap_nm']
  data=dict(radius_min_delta_nm=self.low,radius_max_delta_nm=self.high,bin_width_nm=self.width,archive_native_guard_nm=self.guard,policy='per-radius-bin GEOS overlap and violated-pair champions; native-under25nm challenges immediate, distant champions coalesced over10seconds; archive selection is heuristic and does not discard search proposals',counts=self.counts,pending_contenders=len(self.pending),bins={str(k):v for k,v in sorted(self.bins.items())},pareto_radius_overlap=pareto)
  temporary=self.path/'manifest.tmp.json';temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(self.path/'manifest.json');self.last_write=time.monotonic()
