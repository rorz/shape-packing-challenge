"""Analyze immutable exact-CSP archives; never modifies their solver scripts."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from interval_pair_relation import quotient,intervals,reconstruct,column_orders,verify_element_encoding


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',action='append',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--minimum-cells',type=int,default=100000);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=[];start=time.monotonic()
 for prefix in a.prefix:
  with np.load(prefix+'.domains.npz')as archive:domains=[archive[f'p{i}']for i in range(len(archive.files))]
  with np.load(prefix+'.tables.npz')as archive:
   digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
   if '_domain_digest'in archive.files:assert str(archive['_domain_digest'])==digest
   for key in archive.files:
    if key.startswith('_'):continue
    i,j=map(int,key.split('_'));allowed=archive[key];mask=np.zeros((len(domains[i]),len(domains[j])),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True;row_map,col_map,small=quotient(mask)
    if small.size<a.minimum_cells:continue
    row_reps=np.unique(row_map,return_index=True)[1];col_reps=np.unique(col_map,return_index=True)[1];orders=column_orders(small,domains[i][row_reps],domains[j][col_reps]);metrics=[]
    for name,order in orders.items():
     segments=intervals(small,order);forbidden=intervals(~small,order);counts=np.bincount(segments[:,0],minlength=len(small));metrics.append(dict(order=name,allowed_intervals=len(segments),forbidden_intervals=len(forbidden),mean_per_row=float(counts.mean()),maximum_per_row=int(counts.max())))
    metrics.sort(key=lambda m:(m['allowed_intervals'],m['maximum_per_row']));best=metrics[0];order=orders[best['order']];segments=intervals(small,order);recovered=reconstruct(small.shape,order,segments);assert np.array_equal(recovered,small);assert np.array_equal(recovered[row_map][:,col_map],mask)
    element_cells=verify_element_encoding(mask,row_map,col_map,order,segments)
    basename=Path(prefix).name;np.savez_compressed(a.output/f'{basename}-{key}.intervals.npz',row_map=row_map,column_map=col_map,order=order,intervals=segments,shape=small.shape,domain_digest=np.asarray(digest))
    record=dict(prefix=prefix,pair=[i,j],original_shape=list(mask.shape),quotient_shape=list(small.shape),allowed_tuples=int(small.sum()),forbidden_tuples=int((~small).sum()),best=best,all_orders=metrics,exhaustive_matrix_equivalent=True,exhaustive_actual_element_semantics=True,checked_original_cells=element_cells,tuple_scalar_ratio=float(2*min(small.sum(),(~small).sum())/max(1,3*len(segments))))
    records.append(record);print(json.dumps({k:v for k,v in record.items()if k!='all_orders'}),flush=True)
 result=dict(seconds=time.monotonic()-start,records=records);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
