from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from interval_order_optimization import optimize
from interval_pair_relation import quotient,intervals,reconstruct,verify_element_encoding

out=Path('experiments/2026-11-04/round3/intervals-hamming');out.mkdir(parents=True,exist_ok=True);records=[]
for prefix,pair in [('conflicts/network-angle3',(4,8)),('conflicts/network-angle3',(5,8)),('conflicts-original/wide',(6,10))]:
 path=Path('experiments/2026-11-04/round3')/prefix
 with np.load(path.with_suffix('.domains.npz'))as z:ni,nj=[len(z[f'p{i}'])for i in pair]
 with np.load(path.with_suffix('.tables.npz'))as z:allowed=z[f'{pair[0]}_{pair[1]}']
 mask=np.zeros((ni,nj),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True;rows,cols,small=quotient(mask);start=time.monotonic();order,stats=optimize(small,seconds=2.,iterations=400);segments=intervals(small,order)
 assert stats['final_intervals']==len(segments);assert verify_element_encoding(mask,rows,cols,order,segments)==mask.size;assert np.array_equal(reconstruct(small.shape,order,segments),small)
 np.savez_compressed(out/f'{path.name}-{pair[0]}_{pair[1]}.intervals.npz',row_map=rows,column_map=cols,order=order,intervals=segments,shape=small.shape)
 record=dict(prefix=str(path),pair=pair,shape=small.shape,seconds=time.monotonic()-start,allowed_tuples=int(small.sum()),**stats,exhaustive_actual_element_equivalence=True);records.append(record);print(json.dumps(record),flush=True)
(out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
