"""Exact finite XYZ/angular repair of an invalid saved witness.

The external known-valid engine control is explicit: the invalid witness is
never represented as a feasible positive control. All support cuts allow a
free enclosing center. Failure proves only the saved finite pose domains.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
from experiment_support_bundle import support_bundle
from interval_order_optimization import add_hamming_interval_pair
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load,checked
from arena import check_solution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xy',type=int,default=1);p.add_argument('--angles',type=int,default=20)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--targets',type=float,nargs='+',default=[2.9485269238,2.9485269629])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    raw=Path(a.input).read_bytes();(a.output/'input.json').write_bytes(raw)
    base=load(a.output/'input.json');incumbent=load('solution.json');v=checked(incumbent)
    master.add_pair_constraint=add_hamming_interval_pair
    control_domains=[q[None,:]for q in incumbent]
    control_tables,_=master.exact_tables(control_domains)
    control=master.positive_control(control_domains,control_tables,v.score)
    domains,_=master.correlated_domains(base,directions=0,jitter=a.angles,
                                        minkowski_steps=a.xy,max_domain=10000)
    bundle=support_bundle(domains,base,compute_mec_fast(base))
    (a.output/'bootstrap.json').write_text(json.dumps([master.cut_record(c)for c in bundle],indent=2)+'\n')
    meta=dict(input_sha256=hashlib.sha256(raw).hexdigest(),arguments=vars(a)|{'output':str(a.output)},
              domain_sizes=[len(d)for d in domains],external_control=control,
              witness_valid=checked(base).valid,bundle_count=len(bundle))
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('CONTROL_AND_DOMAIN',json.dumps(meta),flush=True)
    insert=master.add_support_cut
    def expanded(model,variables,radius,cut,serial):
        insert(model,variables,radius,cut,serial)
        if serial==0:
            for k,extra in enumerate(bundle):insert(model,variables,radius,extra,f'bundle_{k}')
    master.add_support_cut=expanded
    tables=None;reports=[]
    for i,target in enumerate(a.targets):
        remaining=a.seconds-(time.monotonic()-started)
        if remaining<1:break
        output=a.output/f'target-{i}.json'
        report=master.solve(domains,target,output,seconds=remaining/(len(a.targets)-i),
                            tables=tables,support_cuts=True,band_screen=True,local_loss=True)
        reports.append(report)
        if output.with_suffix('.tables.npz').exists():
            with np.load(output.with_suffix('.tables.npz'))as z:
                expected=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
                assert str(z['_domain_digest'])==expected
                tables=[(*map(int,k.split('_')),z[k])for k in z.files if not k.startswith('_')]
        for candidate in [output,output.with_suffix('.best-valid.json')]:
            if candidate.exists():
                check=check_solution(json.loads(candidate.read_text()))
                candidate.with_suffix('.api.json').write_text(json.dumps(check,indent=2)+'\n')
                print('API',candidate,json.dumps(check),flush=True)
                if check.get('valid')and check['score']<2.9485269238856113:
                    (a.output/'winner.json').write_bytes(candidate.read_bytes());break
        if (a.output/'winner.json').exists()or report.get('incomplete_tables'):break
    (a.output/'summary.json').write_text(json.dumps(dict(metadata=meta,reports=reports,
        seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()
