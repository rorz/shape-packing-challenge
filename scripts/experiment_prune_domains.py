"""Prune immutable exact CSP archives and retain original categorical indices."""
from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from exact_domain_pruning import IntegerCut,prune,verify_trace


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--domains',required=True);ap.add_argument('--tables',required=True);ap.add_argument('--cuts',action='append',default=[]);ap.add_argument('--target',type=float,default=2.9485269238);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 with np.load(a.domains)as z:domains=[z[f'p{i}']for i in range(len(z.files))]
 digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest();sizes=list(map(len,domains));packed={};pair_sizes=[]
 with np.load(a.tables)as z:
  if'_domain_digest'in z.files:assert str(z['_domain_digest'])==digest
  for key in z.files:
   if key.startswith('_'):continue
   i,j=map(int,key.split('_'));allowed=z[key];mask=np.zeros((sizes[i],sizes[j]),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True;packed[i,j]=np.packbits(mask,axis=1,bitorder='little');packed[j,i]=np.packbits(mask.T,axis=1,bitorder='little');pair_sizes.append([i,j,len(allowed)])
 cuts=[];cut_sources=[]
 for path in a.cuts:
  for k,item in enumerate(json.loads(Path(path).read_text())):
   coeff={int(i):np.asarray(v,dtype=np.int64)for i,v in item['coefficients'].items()};rhs=int(np.floor(a.target*item['scale']))+int(item['offset']);cuts.append(IntegerCut(coeff,rhs));cut_sources.append(dict(path=path,index=k,rhs=rhs,scale=item['scale'],offset=item['offset']))
 loaded=time.monotonic();active,trace,status=prune(sizes,packed,cuts);pruned=time.monotonic();verified=verify_trace(sizes,packed,cuts,active,trace,status);end=time.monotonic();np.savez_compressed(a.output/'original-index-masks.npz',**{f'p{i}':v for i,v in enumerate(active)},**{f'keep{i}':np.flatnonzero(v)for i,v in enumerate(active)})
 (a.output/'elimination-trace.json').write_text(json.dumps(trace)+'\n');(a.output/'cut-sources.json').write_text(json.dumps(cut_sources,indent=2)+'\n')
 result=dict(domains=a.domains,tables=a.tables,cuts=a.cuts,domain_digest=digest,target=a.target,initial_sizes=sizes,remaining_sizes=[int(v.sum())for v in active],removed=sum(sizes)-sum(int(v.sum())for v in active),status=status,eliminations_replayed_exactly=verified,cut_count=len(cuts),directed_pair_count=len(packed),seconds=dict(load=loaded-start,prune=pruned-loaded,verify=end-pruned,total=end-start),scope='Exact finite domains and supplied integer cuts only; original pose IDs retained')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
