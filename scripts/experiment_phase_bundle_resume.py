"""Cached phase-pool solve with all balanced supports at every physical phase."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
import experiment_support_bundle as bundled


def all_phase_supports(domains,anchors,circles):
    result=[];seen=set()
    for phase,(anchor,circle)in enumerate(zip(anchors,circles)):
        for cut in bundled.support_bundle(domains,anchor,circle):
            digest=hashlib.sha256(str(cut['offset']).encode()+b''.join(str(i).encode()+v.tobytes()
                for i,v in sorted(cut['coefficients'].items()))).hexdigest()
            if digest in seen:continue
            seen.add(digest);cut['anchor_phase_index']=phase;result.append(cut)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',required=True)
    parser.add_argument('--tables',required=True)
    parser.add_argument('--anchors',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--target',type=float,default=2.9485269238)
    parser.add_argument('--seconds',type=float,default=300)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--expected-score',type=float)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--hamming',action='store_true')
    parser.add_argument('--masks')
    parser.add_argument('--masks-target',type=float)
    args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Use1 through4 workers')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    archive=np.load(args.domains);domains=[archive[f'p{i}']for i in range(len(archive.files))]
    archive=np.load(args.anchors);anchors,circles=archive['anchors'],archive['anchor_circles']
    started=time.monotonic();cuts=all_phase_supports(domains,anchors,circles)
    count=sum(v.size for c in cuts for v in c['coefficients'].values())
    metadata=dict(arguments=vars(args),bundle_count=len(cuts),coefficient_count=count,
                  coefficient_bytes=sum(v.nbytes for c in cuts for v in c['coefficients'].values()),
                  build_seconds=time.monotonic()-started,domain_sizes=[len(d)for d in domains],
                  domain_sha256=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest(),
                  scope='Same finite poses/pair masks; strictly stronger valid radius inequalities, never global infeasibility')
    output.with_suffix('.preflight.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PREFLIGHT',json.dumps(metadata),flush=True)
    if args.preflight_only:
        output.with_suffix('.all-cuts.json').write_text(json.dumps([master.cut_record(c)for c in cuts],indent=2)+'\n')
        return
    archive=np.load(args.tables)
    tables=[(*map(int,k.split('_')),archive[k])for k in archive.files if not k.startswith('_')]
    if '_domain_digest'in archive.files and str(archive['_domain_digest'])!=metadata['domain_sha256']:
        raise ValueError('Table domain digest mismatch')
    original_bundle=bundled.support_bundle
    original_solver=master.cp_model.CpSolver
    original_insert=master.add_support_cut
    if args.masks:
        if args.masks_target!=args.target:
            raise ValueError('Masks must be proved at this exact target, not a stricter bound')
        archive=np.load(args.masks);masks=[np.asarray(archive[f'p{i}'],dtype=bool)for i in range(len(domains))]
        if any(m.shape!=(len(d),)for m,d in zip(masks,domains)):raise ValueError('Mask shape mismatch')
        metadata['mask_counts']=[int(m.sum())for m in masks]
        metadata['mask_sha256']=hashlib.sha256(Path(args.masks).read_bytes()).hexdigest()
        metadata['mask_scope']='Target model only; incumbent positive control uses original unpruned domains'
        output.with_suffix('.preflight.json').write_text(json.dumps(metadata,indent=2)+'\n')
        def insert_masks(model,variables,radius,cut,serial):
            original_insert(model,variables,radius,cut,serial)
            if serial==0:
                for variable,mask in zip(variables,masks):
                    model.add_allowed_assignments([variable],np.flatnonzero(mask)[:,None].tolist())
        master.add_support_cut=insert_masks
    trace=[];solve_started=time.monotonic()
    class TracedSolver(original_solver):
        def solve(self,model,solution_callback=None):
            control=model.proto.variables[0].name.startswith('control_')
            self.parameters.num_search_workers=1 if control else args.workers
            if not control:
                self.parameters.max_time_in_seconds=max(.01,args.seconds-(time.monotonic()-solve_started))
                self.parameters.stop_after_first_solution=True
            before=time.monotonic()
            status=super().solve(model,solution_callback)
            record=dict(control=control,status=self.status_name(status),seconds=time.monotonic()-before,
                        workers=self.parameters.num_search_workers,stats=self.response_stats())
            trace.append(record);output.with_suffix('.cp-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
            print('CP_RESPONSE',json.dumps({k:v for k,v in record.items()if k!='stats'}),flush=True)
            return status
    bundled.support_bundle=lambda *unused:cuts
    master.cp_model.CpSolver=TracedSolver
    restore_interval=None
    if args.hamming:
        import interval_pair_relation as relation
        from interval_order_optimization import add_hamming_interval_pair
        restore_interval=relation.add_allowed_interval_pair;relation.add_allowed_interval_pair=add_hamming_interval_pair
    try:
        bundled.solve_bundle(domains,tables,args.target,output,args.seconds,args.expected_score,use_intervals=True)
    finally:
        bundled.support_bundle=original_bundle;master.cp_model.CpSolver=original_solver
        master.add_support_cut=original_insert
        if restore_interval is not None:relation.add_allowed_interval_pair=restore_interval


if __name__=='__main__':main()
