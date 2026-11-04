"""Mix locally repaired poses from many small rigid rotations in one exact CSP.

The stored joint moves are rounded phase atoms, not an LLL basis. A rotation
about piece zero preserves its XY gauge. Full positive/negative physical phase
anchors also supply balanced radius supports, avoiding a fixed-direction
quadratic underestimate across the phase interval.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
import experiment_support_bundle as bundled
from conflict_area_band import true_area_loss_band
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import load


def phase_domains(base,phases=25,radius=300e-6,xy_steps=1,jitter=3,max_domain=1600):
    if phases<3 or phases%2!=1:raise ValueError('Use an odd phase count >=3')
    base=np.round(base,6);pivot=base[0,:2].copy();circle=np.asarray(compute_mec_fast(base))
    angles=np.linspace(-radius,radius,phases)
    anchors=[];circles=[]
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        p=base.copy();p[:,:2]=(p[:,:2]-pivot)@rotation.T+pivot;p[:,2]+=angle
        anchors.append(p)
        circles.append(np.r_[(circle[:2]-pivot)@rotation.T+pivot,circle[2]])
    anchors=np.asarray(anchors);circles=np.asarray(circles)
    positive=anchors[len(angles)//2+1:]
    moves=np.rint((positive-base)*1e6)
    assert np.all(moves[:,0,:2]==0)
    domains,selected=master.correlated_domains(base,moves,directions=len(moves),jitter=jitter,
                                               minkowski_steps=xy_steps,max_domain=max_domain)
    # Explicitly verify coverage, including negative atoms' second-order error.
    keys=[{tuple(np.rint(p*1e6).astype(np.int64))for p in d}for d in domains]
    for anchor in anchors:
        grid=np.round(anchor,6);grid[:,2]=[canonicalize_theta(t)for t in grid[:,2]]
        for i,p in enumerate(grid):
            if tuple(np.rint(p*1e6).astype(np.int64))not in keys[i]:
                raise ValueError('Requested physical rounded phase is outside generated local atoms')
    return domains,anchors,circles,moves,angles,selected


def estimate(domains,seed=110451):
    rng=np.random.default_rng(seed);raw=projected=0.;details=[]
    for i,j in itertools.combinations(range(len(domains)),2):
        a,b=domains[i],domains[j]
        if master.separated_domains(a,b):continue
        ia=rng.choice(len(a),min(96,len(a)),replace=False);ib=rng.choice(len(b),min(96,len(b)),replace=False)
        area,loss=true_area_loss_band(a[ia],b[ib])
        ambiguous=(area>1e-6-1e-9)&(area<=1e-6+loss+1e-9)
        size=len(a)*len(b);amount=size*ambiguous.mean();raw+=size;projected+=amount
        details.append(dict(pair=[i,j],raw=size,sample=area.size,ambiguous=int(ambiguous.sum()),projected_geos=float(amount)))
    return dict(seed=seed,raw_pairs=int(raw),estimated_geos=float(projected),details=details,
                scope='Random row/column point estimate only; full decisions computed in actual tables')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--phases',type=int,default=25)
    parser.add_argument('--range-micro',type=float,default=300)
    parser.add_argument('--jitter',type=int,default=3)
    parser.add_argument('--xy-steps',type=int,default=1)
    parser.add_argument('--max-domain',type=int,default=1600)
    parser.add_argument('--seconds',type=float,default=450)
    parser.add_argument('--target',type=float,default=2.9485269238)
    parser.add_argument('--expected-score',type=float)
    parser.add_argument('--estimate-only',action='store_true')
    parser.add_argument('--hamming',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    base=load(args.input)
    domains,anchors,circles,moves,angles,selected=phase_domains(base,args.phases,args.range_micro*1e-6,
        args.xy_steps,args.jitter,args.max_domain)
    np.savez_compressed(output.with_suffix('.atoms.npz'),placements=base,origin=base,moves=moves,
                         anchors=anchors,anchor_circles=circles,phases=angles)
    metadata=dict(arguments=vars(args),input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        atom_kind='Rounded rigid phase displacements, not a lattice basis',selected_indices=selected,
        domain_sizes=[len(d)for d in domains],gauge='All rigid rotations about piece zero; zero common XY displacement',
        rounded_physical_phase_coverage=True)
    output.with_suffix('.input.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PHASE_POOL',json.dumps(metadata),flush=True)
    if args.estimate_only:
        result=estimate(domains);output.with_suffix('.estimate.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ESTIMATE',json.dumps({k:v for k,v in result.items()if k!='details'}),flush=True)
        return
    original_bundle=bundled.support_bundle
    def phase_bundle(dd,p,c):
        cuts=original_bundle(dd,p,c);seen=set()
        def key(cut):
            return hashlib.sha256(str(cut['offset']).encode()+b''.join(str(i).encode()+v.tobytes()
                for i,v in sorted(cut['coefficients'].items()))).hexdigest()
        for cut in cuts:seen.add(key(cut))
        for anchor,circle in zip(anchors,circles):
            cut=master.categorical_support_cut(dd,anchor,circle)
            if cut is not None and key(cut)not in seen:
                seen.add(key(cut));cuts.append(cut)
        return cuts
    bundled.support_bundle=phase_bundle
    restore_interval=None
    if args.hamming:
        import interval_pair_relation as relation
        from interval_order_optimization import add_hamming_interval_pair
        restore_interval=relation.add_allowed_interval_pair
        relation.add_allowed_interval_pair=add_hamming_interval_pair
    try:
        bundled.solve_bundle(domains,None,args.target,output,args.seconds,args.expected_score,
            use_intervals=True,band_screen=True,local_loss=True)
    finally:
        bundled.support_bundle=original_bundle
        if restore_interval is not None:relation.add_allowed_interval_pair=restore_interval


if __name__=='__main__':main()
