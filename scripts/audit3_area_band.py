"""Audit mathematical true-area/polygon-loss screens against retained GEOS tables."""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from experiment_conflict_master import POLYGON_LOSS,separated_domains
from experiment_contact_csp import pair_kernel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--guard',type=float,default=1e-9)
    parser.add_argument('--local-loss',action='store_true')
    args=parser.parse_args()
    kernel=pair_kernel();start=time.monotonic();seen=set();results=[]
    for table_path in sorted(args.directory.glob('*.tables.npz')):
        domain_path=table_path.with_name(table_path.name.replace('.tables.npz','.domains.npz'))
        archive=np.load(domain_path);domains=[archive[f'p{i}']for i in range(len(archive.files))]
        archive=np.load(table_path);tables={tuple(map(int,k.split('_'))):archive[k]for k in archive.files if not k.startswith('_')}
        digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)+b''.join(v.tobytes()for v in tables.values())).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        record=dict(domains=str(domain_path),tables=str(table_path),semantic_hash=digest,
                    entries=0,screen_allowed=0,screen_rejected=0,ambiguous=0,false_allow=0,false_reject=0)
        for i,j in itertools.combinations(range(len(domains)),2):
            a,b=domains[i],domains[j]
            if separated_domains(a,b):continue
            if args.local_loss:
                from conflict_area_band import true_area_loss_band
                true,loss=true_area_loss_band(a,b)
            else:
                true=np.empty((len(a),len(b)))
                kernel(len(a),len(b),np.ascontiguousarray(a),np.ascontiguousarray(b),true)
                loss=POLYGON_LOSS
            accepted=true<=1e-6-args.guard
            rejected=true>1e-6+loss+args.guard
            mask=np.ones_like(accepted)if(i,j)not in tables else np.zeros_like(accepted)
            if(i,j)in tables:
                allowed=tables[i,j];mask[allowed[:,0],allowed[:,1]]=True
            record['entries']+=true.size
            record['screen_allowed']+=int(accepted.sum())
            record['screen_rejected']+=int(rejected.sum())
            record['ambiguous']+=int((~(accepted|rejected)).sum())
            record['false_allow']+=int((accepted&~mask).sum())
            record['false_reject']+=int((rejected&mask).sum())
        results.append(record)
        print(json.dumps(record),flush=True)
    result=dict(guard=args.guard,polygon_loss_bound=float(POLYGON_LOSS),local_loss=args.local_loss,cases=results,
                seconds=time.monotonic()-start,scope='Mathematical polygon deficit band; native arithmetic allowance is conservative numerical guard, not an interval arithmetic certificate')
    result['passed']=all(not r['false_allow']and not r['false_reject']for r in results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',result['passed'],'cases',len(results),'seconds',result['seconds'],flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
