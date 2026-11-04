"""Sound fixed-point pruning of finite categorical domains using integer data."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class IntegerCut:
 coefficients:dict[int,np.ndarray]
 rhs:int


def pack_relations(masks):
 result={}
 for (i,j),mask in masks.items():
  result[i,j]=np.packbits(mask,axis=1,bitorder='little')
  result[j,i]=np.packbits(mask.T,axis=1,bitorder='little')
 return result


def prune(sizes,relations,cuts):
 """Retain original pose IDs; each removed value has a replayable reason."""
 active=[np.ones(n,dtype=bool)for n in sizes];neighbors=[set()for _ in sizes];watch=[set()for _ in sizes]
 for i,j in relations:neighbors[j].add(i)
 for k,cut in enumerate(cuts):
  for i,coeff in cut.coefficients.items():
   if len(coeff)!=sizes[i]:raise ValueError('Cut does not match original pose IDs')
   watch[i].add(k)
 arcs=deque(relations);arc_set=set(relations);pending=deque(range(len(cuts)));cut_set=set(range(len(cuts)));trace=[];counts=dict(arc_revisions=0,cut_revisions=0,pair_removed=0,cut_removed=0)
 def changed(i):
  for k in neighbors[i]:
   if(k,i)not in arc_set:arcs.append((k,i));arc_set.add((k,i))
  for k in watch[i]:
   if k not in cut_set:pending.append(k);cut_set.add(k)
 def remove(i,drop,reason):
  values=np.flatnonzero(drop&active[i]);
  if not len(values):return False
  active[i][values]=False;trace.append(dict(piece=i,removed=values.tolist(),**reason));changed(i)
  return True
 while pending or arcs:
  if any(not mask.any()for mask in active):return active,trace,dict(infeasible=True,reason='empty_domain',counts=counts)
  if pending:
   k=pending.popleft();cut_set.remove(k);cut=cuts[k];counts['cut_revisions']+=1
   minima={i:int(np.min(coeff[active[i]]))for i,coeff in cut.coefficients.items()};total=sum(minima.values())
   if total>cut.rhs:return active,trace,dict(infeasible=True,reason='cut_lower_bound',cut=k,total_minimum=total,rhs=cut.rhs,counts=counts)
   for i,coeff in cut.coefficients.items():
    upper=cut.rhs-total+minima[i];drop=active[i]&(coeff>upper);n=int(drop.sum())
    if remove(i,drop,dict(kind='cut',cut=k,upper=upper)):counts['cut_removed']+=n
  else:
   i,j=arcs.popleft();arc_set.remove((i,j));counts['arc_revisions']+=1
   support=np.bitwise_and(relations[i,j],np.packbits(active[j],bitorder='little')).any(axis=1);drop=active[i]&~support;n=int(drop.sum())
   if remove(i,drop,dict(kind='pair',other=j)):counts['pair_removed']+=n
 return active,trace,dict(infeasible=False,fixed_point=True,counts=counts)


def verify_trace(sizes,relations,cuts,active,trace,status):
 """Independent replay checks every elimination against its contemporaneous state."""
 replay=[np.ones(n,dtype=bool)for n in sizes]
 for event in trace:
  i=event['piece'];removed=np.asarray(event['removed'],dtype=int);assert replay[i][removed].all()
  if event['kind']=='pair':
   j=event['other'];supported=np.bitwise_and(relations[i,j][removed],np.packbits(replay[j],bitorder='little')).any(axis=1);assert not supported.any()
  else:
   cut=cuts[event['cut']];other_min=sum(int(v[replay[k]].min())for k,v in cut.coefficients.items()if k!=i);assert all(int(v)+other_min>cut.rhs for v in cut.coefficients[i][removed])
  replay[i][removed]=False
 assert all(np.array_equal(a,b)for a,b in zip(active,replay))
 if status['infeasible']:
  if status['reason']=='empty_domain':assert any(not p.any()for p in replay)
  else:
   cut=cuts[status['cut']];assert sum(int(v[replay[i]].min())for i,v in cut.coefficients.items())>cut.rhs
 return True
