"""Target-UNSAT assumptions over exact pair relations and radius inequalities.

Only cached finite domains/tables are analyzed. Conditional constraints do not
disable their total-function definitions. A reported core is sufficient, not
necessarily minimum; expansion is proposed only after inspecting its pieces.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
import experiment_conflict_master as master
from experiment_support_bundle import support_bundle
from interval_pair_relation import add_allowed_interval_pair
from separable_support import Inserter
from semicircle_packing.polish import load,checked,save
from semicircle_packing.fast_mec import compute_mec_fast


class AssumedModel:
    def __init__(self,domains,target):
        self.domains=domains;self.model=cp_model.CpModel()
        self.variables=[self.model.new_int_var(0,len(d)-1,f'piece_{i}')for i,d in enumerate(domains)]
        self.radius=self.model.new_int_var(0,int(np.floor(target*1e12)),'radius_bound')
        self.literals=[];self.labels=[];self.support=Inserter(domains)
    def literal(self,label):
        b=self.model.new_bool_var(f'assumption_{len(self.literals)}')
        self.literals.append(b);self.labels.append(label);return b
    def pair(self,i,j,allowed):
        enabled=self.literal(dict(kind='pair',pair=[i,j],pieces=[i,j]))
        original=self.model.add_allowed_assignments
        def conditional(*args,**kwargs):
            constraint=original(*args,**kwargs);constraint.only_enforce_if(enabled);return constraint
        self.model.add_allowed_assignments=conditional
        try:add_allowed_interval_pair(self.model,self.variables[i],self.variables[j],allowed,len(self.domains[i]),len(self.domains[j]))
        finally:self.model.add_allowed_assignments=original
    def cut(self,cut,index):
        enabled=self.literal(dict(kind='support',cut=index,pieces=sorted(map(int,cut['coefficients']))))
        original=self.model.add;constraints=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);constraints.append(result);return result
        self.model.add=capture
        try:self.support(self.model,self.variables,self.radius,cut,index)
        finally:self.model.add=original
        if not constraints:raise AssertionError('Support inserted no inequality')
        constraints[-1].only_enforce_if(enabled)
    def solve(self,indices,seconds):
        self.model.clear_assumptions();self.model.add_assumptions([self.literals[i]for i in indices])
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
        solver.parameters.max_time_in_seconds=max(.01,seconds)
        solver.parameters.stop_after_first_solution=True
        if hasattr(solver.parameters,'core_minimization_level'):solver.parameters.core_minimization_level=2
        start=time.monotonic();status=solver.solve(self.model)
        result=dict(status=solver.status_name(status),seconds=time.monotonic()-start,assumptions=len(indices),stats=solver.response_stats())
        if status==cp_model.INFEASIBLE:
            literal_ids=set(solver.sufficient_assumptions_for_infeasibility())
            result['core']=[i for i,b in enumerate(self.literals)if b.index in literal_ids]
        elif status in(cp_model.OPTIMAL,cp_model.FEASIBLE):
            result['assignment']=[solver.value(v)for v in self.variables]
        return result


def load_cuts(path):
    result=[]
    for c in json.loads(Path(path).read_text()):
        c['coefficients']={int(i):np.asarray(v,dtype=np.int64)for i,v in c['coefficients'].items()}
        result.append(c)
    return result


def build(domains,tables,cuts,target):
    assumed=AssumedModel(domains,target)
    for i,j,allowed in tables:assumed.pair(i,j,allowed)
    for index,cut in enumerate(cuts):assumed.cut(cut,index)
    return assumed


def pieces(labels,indices):return sorted(set(p for i in indices for p in labels[i]['pieces']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',required=True);parser.add_argument('--tables',required=True)
    parser.add_argument('--cuts',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--target',type=float,default=2.9485269238806113)
    parser.add_argument('--seconds',type=float,default=90)
    parser.add_argument('--control',default='experiments/2026-11-04/round3/conflicts/network-input/incumbent.json')
    args=parser.parse_args();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    archive=np.load(args.domains);domains=[archive[f'p{i}']for i in range(len(archive.files))]
    archive=np.load(args.tables);tables=[(*map(int,k.split('_')),archive[k])for k in archive.files if not k.startswith('_')]
    digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
    if '_domain_digest'in archive.files and str(archive['_domain_digest'])!=digest:raise ValueError('Domain digest mismatch')
    cuts=load_cuts(args.cuts)
    # Explicit external control through this conditional encoder.
    control=load(args.control);validation=checked(control)
    if not validation.valid:raise ValueError('External control is invalid')
    cd=[q[None]for q in control];ct,_=master.exact_tables(cd,band_screen=True,local_loss=True)
    cc=support_bundle(cd,control,validation.mec);cm=build(cd,ct,cc,validation.score+1e-10)
    control_result=cm.solve(list(range(len(cm.labels))),10)
    if control_result['status']not in('OPTIMAL','FEASIBLE'):raise AssertionError('Assumption encoder rejects external control')
    start=time.monotonic();deadline=start+args.seconds;assumed=build(domains,tables,cuts,args.target)
    report=dict(target=args.target,domain_sizes=[len(d)for d in domains],domain_digest=digest,
        external_control=dict(input=args.control,score=validation.score,result=control_result),labels=assumed.labels,
        scope='Cached finite domains, conditional pair relations and valid radius bounds; sufficient core only')
    current=list(range(len(assumed.labels)));initial=assumed.solve(current,min(45,deadline-time.monotonic()))
    report['initial']=initial;print('INITIAL',json.dumps({k:v for k,v in initial.items()if k!='stats'}),flush=True)
    if initial['status']in('OPTIMAL','FEASIBLE'):
        candidate=np.array([d[k]for d,k in zip(domains,initial['assignment'])]);save(output.with_suffix('.candidate.json'),candidate)
        v=checked(candidate);report['candidate']=dict(valid=v.valid,score=v.score,errors=v.errors,mec=compute_mec_fast(candidate))
    elif initial['status']=='INFEASIBLE':
        current=initial['core'];report['initial_pieces']=pieces(assumed.labels,current);steps=[]
        # First attempt removing an entire piece's constraints: localize geometry,
        # rather than merely minimize the number of assumptions.
        candidates=pieces(assumed.labels,current)
        candidates.sort(key=lambda p:sum(p in assumed.labels[i]['pieces']for i in current))
        for p in candidates:
            if time.monotonic()>=deadline:break
            trial=[i for i in current if p not in assumed.labels[i]['pieces']]
            if len(trial)==len(current):continue
            result=assumed.solve(trial,min(3,deadline-time.monotonic()))
            steps.append(dict(remove_piece=p,status=result['status'],seconds=result['seconds']))
            if result['status']=='INFEASIBLE':current=result['core']
        for i in list(current):
            if time.monotonic()>=deadline:break
            if i not in current:continue
            result=assumed.solve([j for j in current if j!=i],min(2,deadline-time.monotonic()))
            steps.append(dict(remove_assumption=i,status=result['status'],seconds=result['seconds']))
            if result['status']=='INFEASIBLE':current=result['core']
        report.update(core=current,core_labels=[assumed.labels[i]for i in current],critical_pieces=pieces(assumed.labels,current),
                      shrink_steps=steps,minimum_core=False,localized=len(pieces(assumed.labels,current))<=7)
    report['seconds']=time.monotonic()-start
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',json.dumps({k:v for k,v in report.items()if k not in('labels','external_control','initial','shrink_steps')}),flush=True)


if __name__=='__main__':main()
