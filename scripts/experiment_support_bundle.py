"""Bootstrap the finite master with every baseline balanced support inequality.

This standalone prototype leaves experiment_conflict_master.py unchanged. It
temporarily wraps its model-insertion hook inside this process, adding the
retained bundle both to the forced incumbent control and to the target model.
Full placement variables and exact pair relations are unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
from experiment_hillclimb import dual_weights
from semicircle_packing.fast_mec import compute_mec_fast


def support_bundle(domains,base,circle):
    reference=master.categorical_support_cut(domains,base,circle)
    if reference is None:return []
    _,weights=dual_weights(base,np.asarray(circle[:2]),circle[2])
    pieces=np.asarray(reference['direction_pieces'])
    directions=np.asarray(reference['directions'])
    assert weights.shape[0]==len(pieces)
    anchor=np.asarray(reference['anchor'])
    supports=[]
    for i,u in zip(pieces,directions):
        p=domains[i];co,sn=np.cos(p[:,2]),np.sin(p[:,2])
        unit=np.where(co*u[0]+sn*u[1]>=0,1.,np.abs(-sn*u[0]+co*u[1]))
        supports.append((p[:,:2]-anchor)@u+unit)
    result=[];seen=set();scale=reference['scale'];center_bound=reference['center_bound']
    for ww in weights.T:
        contributions={}
        for i,w,h in zip(pieces,ww,supports):
            if w<=0:continue
            i=int(i)
            contributions[i]=contributions.get(i,np.zeros(len(domains[i])))+w*h
        coefficients={i:np.floor(h*scale).astype(np.int64)for i,h in contributions.items()}
        residual=float(np.linalg.norm(directions.T@ww))
        offset=int(np.ceil((center_bound*residual+1e-11)*scale))
        fingerprint=hashlib.sha256(str(offset).encode()+b''.join(
            str(i).encode()+h.tobytes()for i,h in sorted(coefficients.items()))).hexdigest()
        if fingerprint in seen:continue
        seen.add(fingerprint)
        cut=dict(reference,coefficients=coefficients,offset=offset,residual=residual,
                 weights=ww.tolist(),lower_bound=float(sum(w*h[0]for w,h in zip(ww,supports))-center_bound*residual))
        result.append(cut)
    return result


def solve_bundle(domains,tables,target,output,seconds=180,expected_score=None,seed=110443,
                 use_intervals=False,band_screen=False,local_loss=False):
    base=np.asarray([d[0]for d in domains]);circle=compute_mec_fast(base)
    bundle=support_bundle(domains,base,circle)
    prefix=Path(output);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.bootstrap-cuts.json').write_text(json.dumps([master.cut_record(c)for c in bundle],indent=2)+'\n')
    metadata=dict(bundle_count=len(bundle),base_circle=circle,intervals=use_intervals,
                  domain_sizes=[len(d)for d in domains],target=target,
                  scope='Every retained baseline balanced support inequality; exact finite domains only')
    prefix.with_suffix('.bootstrap.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('BOOTSTRAP',json.dumps(metadata),flush=True)
    insert=master.add_support_cut
    def insert_bundle(model,variables,radius,cut,serial):
        insert(model,variables,radius,cut,serial)
        if serial==0 or serial=='control':
            for k,additional in enumerate(bundle):
                insert(model,variables,radius,additional,f'bundle_{serial}_{k}')
            print('BUNDLE_INSERTED',serial,len(bundle),flush=True)
    master.add_support_cut=insert_bundle
    pair_insert=master.add_pair_constraint
    if use_intervals:
        from interval_pair_relation import add_allowed_interval_pair
        master.add_pair_constraint=add_allowed_interval_pair
    try:
        return master.solve(domains,target,output,seconds=seconds,seed=seed,tables=tables,
                            support_cuts=True,check_positive=True,expected_score=expected_score,
                            band_screen=band_screen,local_loss=local_loss)
    finally:
        master.add_support_cut=insert
        master.add_pair_constraint=pair_insert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains')
    parser.add_argument('--tables')
    parser.add_argument('--input')
    parser.add_argument('--basis')
    parser.add_argument('--directions',type=int,default=10)
    parser.add_argument('--jitter',type=int,default=3)
    parser.add_argument('--minkowski-steps',type=int,default=1)
    parser.add_argument('--max-domain',type=int,default=5000)
    parser.add_argument('--pair-sums',action='store_true')
    parser.add_argument('--intervals',action='store_true')
    parser.add_argument('--band-screen',action='store_true')
    parser.add_argument('--local-loss',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--target',type=float,default=2.9485269238)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--expected-score',type=float)
    parser.add_argument('--seed',type=int,default=110443)
    args=parser.parse_args()
    if args.domains:
        archive=np.load(args.domains);domains=[archive[f'p{i}']for i in range(len(archive.files))]
        selected=[]
    elif args.input:
        from semicircle_packing.polish import load
        base=load(args.input)
        moves=np.load(args.basis)['moves']if args.basis else None
        domains,selected=master.correlated_domains(base,moves,directions=args.directions,jitter=args.jitter,
            pair_sums=args.pair_sums,minkowski_steps=args.minkowski_steps,max_domain=args.max_domain)
    else:parser.error('--domains or --input required')
    tables=None
    if args.tables:
        archive=np.load(args.tables)
        tables=[(*map(int,k.split('_')),archive[k])for k in archive.files if not k.startswith('_')]
        if '_domain_digest'in archive.files:
            digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
            if str(archive['_domain_digest'])!=digest:raise ValueError('Tables do not match domain digest')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).with_suffix('.input.json').write_text(json.dumps(dict(arguments=vars(args),selected_basis_indices=selected),indent=2)+'\n')
    solve_bundle(domains,tables,args.target,args.output,args.seconds,args.expected_score,args.seed,
                 use_intervals=args.intervals,band_screen=args.band_screen,local_loss=args.local_loss)


if __name__=='__main__':main()
