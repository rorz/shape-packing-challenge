"""Exact selected-cell pair oracle and free-center cuts for one finite phase."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
import experiment_conflict_master as master
from lazy_pair_oracle import PairOracle
from interval_order_optimization import add_hamming_interval_pair
from separable_support import Inserter,audit,factor,piece_values
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import checked,save,load
from semicircle_packing.scoring import compute_mec
from semicircle_packing.geometry import Semicircle,canonicalize_theta
from arena import check_solution

LEADER=2.9485269238856113
ACCEPTANCE_GUARD=2e-12


def submission_for_geometry(placements):
 """Encode scored grid poses as raw input without normalizing them twice.

 The two scored angles just outside +/-pi require a raw +/-3pi alias.
 Normalizing those scored angles again changes the pair-oracle geometry.
 """
 p=np.asarray(placements,dtype=float);raw=np.round(p,6)
 if np.max(np.abs(raw-p))>1e-12:raise ValueError('Finite phase domains must use six-decimal scored poses')
 for i,theta in enumerate(raw[:,2]):
  if canonicalize_theta(theta)!=theta:
   match=next((v for v in(-9.424778,9.424778)if canonicalize_theta(v)==theta),None)
   if match is None:raise ValueError(f'No raw submission encoding for scored angle {theta}')
   raw[i,2]=match
 scored=raw.copy();scored[:,2]=[canonicalize_theta(t)for t in raw[:,2]]
 if not np.allclose(scored,p,rtol=0,atol=1e-12):raise ValueError('Submission encoding changed the finite-domain geometry')
 return raw,scored


def solve_phase(domains,base,target,output,*,seconds=15,seed=3110811,control=False,local_only=False,incumbent_score=None):
 """Solve a finite placement neighborhood.

 ``local_only`` and ``control`` are offline feasibility modes: neither reads
 the working-tree incumbent nor calls the remote API. They accept a leaf only
 after reference validation at radius <= target + 2e-12. Returned assignments
 index the supplied domains; solution.json contains raw submission poses.
 No held-out solution is supplied as a hint. Default search mode retains API
 checks of incumbent improvements; incumbent_score avoids its global-file read.
 """
 output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+seconds;np.savez_compressed(output/'domains.npz',**{f'p{i}':d for i,d in enumerate(domains)})
 offline=local_only or control;best=incumbent_score
 if best is None and not offline:best=checked(load('solution.json')).score
 if best is not None:best=float(best)
 try:oracle=PairOracle(domains,deadline=deadline)
 except TimeoutError:
  result=dict(valid=False,status='PREPARATION_DEADLINE',seconds=time.monotonic()-start,scope='Incomplete conservative relaxation; no feasibility conclusion');(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
 model=cp_model.CpModel();variables=[model.new_int_var(0,len(d)-1,f'pose_{i}')for i,d in enumerate(domains)]
 for i,j,allowed in oracle.relaxed_tables():add_hamming_interval_pair(model,variables[i],variables[j],allowed,len(domains[i]),len(domains[j]))
 circle=compute_mec_fast(base);cuts=__import__('experiment_support_bundle').support_bundle(domains,base,circle);support_audit=audit(domains,cuts);insert=Inserter(domains);radius=model.new_int_var(0,int(np.floor(target*1e12)),'radius_lower_bound')
 for k,c in enumerate(cuts):insert(model,variables,radius,c,k)
 model.minimize(radius)
 for v in variables:model.add_hint(v,0)
 (output/'initial-support-audit.json').write_text(json.dumps(support_audit,indent=2)+'\n');events=output/'events.jsonl';counts=dict(iterations=0,pair_nogoods=0,enclosure_cuts=0,enclosure_nogoods=0,reference_checks=0);result=dict(valid=False,domain_sizes=list(map(len,domains)),target=target,acceptance_guard=ACCEPTANCE_GUARD,local_only=offline,initial_band_seconds=time.monotonic()-start,support_cut_count=len(cuts),relaxation='Every conservative-band uncertain cell is allowed until selected GEOS resolves it')
 def event(kind,**data):
  with events.open('a')as f:f.write(json.dumps(dict(event=kind,seconds=time.monotonic()-start,**data))+'\n')
 while time.monotonic()<deadline:
  solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=deadline-time.monotonic();solver.parameters.num_search_workers=1;solver.parameters.random_seed=seed+counts['iterations'];solver.parameters.stop_after_first_solution=True;status=solver.solve(model);counts['iterations']+=1
  if status==cp_model.INFEASIBLE:result.update(status='INFEASIBLE',finite_exhausted=True);break
  if status not in(cp_model.OPTIMAL,cp_model.FEASIBLE):result['status']=solver.status_name(status);break
  assignment=[solver.value(v)for v in variables];bad=oracle.check(assignment)
  if bad:
   for conflict in bad:
    i,j=conflict['pair'];model.add_forbidden_assignments([variables[i],variables[j]],[conflict['values']]);counts['pair_nogoods']+=1
   event('pair_conflicts',assignment=assignment,conflicts=bad)
  else:
   p=np.asarray([d[value]for d,value in zip(domains,assignment)]);circle=compute_mec_fast(p);event('exact_pair_feasible',assignment=assignment,circle=circle)
   if offline or circle[2]<best-ACCEPTANCE_GUARD or circle[2]<=target+ACCEPTANCE_GUARD:
    raw,scored=submission_for_geometry(p);validation=checked(raw);counts['reference_checks']+=1
    if offline and validation.valid and validation.score<=target+ACCEPTANCE_GUARD:
     save(output/'solution.json',raw);result.update(status='CONTROL_FEASIBLE'if control else'LOCAL_FEASIBLE',valid=True,score=float(validation.score),assignment=assignment,raw_placements=raw.tolist(),scored_placements=scored.tolist(),solution_path=str(output/'solution.json'));break
    if not offline and validation.valid and validation.score<best-ACCEPTANCE_GUARD:
     save(output/'best.json',raw);api=check_solution(json.loads((output/'best.json').read_text()));(output/'best-api.json').write_text(json.dumps(api,indent=2)+'\n');event('api',response=api)
     if api.get('valid'):best=float(api['score']);result.update(best_score=best,api=api)
     if api.get('valid')and api['score']<LEADER:result.update(valid=True,status='SERVER_FIRST_PLACE',score=api['score']);break
   cut=master.categorical_support_cut(domains,p,circle)
   if cut is not None:insert(model,variables,radius,cut,f'learned_{counts["enclosure_cuts"]}');counts['enclosure_cuts']+=1
   # If the conservative support cut does not reject this leaf, use the
   # existing independent point-support conflict, or a checked full leaf.
   factored=None if cut is None else factor(cut,insert.prepared)
   rejects=False if factored is None else sum(int(piece_values(factored,insert.prepared,i)[assignment[i]])for i in factored['pieces'])>int(np.floor(target*1e12))*factored['ratio']+factored['offset']
   if not rejects:
    owners,bound=master.boundary_conflict(p,circle[:2],target)
    if owners is None:
     reference=compute_mec([Semicircle(*v)for v in p])
     if reference[2]<=target+2e-12:result.update(status='NUMERIC_ENCLOSURE_BAND',reference_mec=reference);break
     owners=list(range(len(domains)))
    model.add_forbidden_assignments([variables[i]for i in owners],[[assignment[i]for i in owners]]);counts['enclosure_nogoods']+=1
  if hasattr(model,'clear_hints'):model.clear_hints()
  for variable,value in zip(variables,assignment):model.add_hint(variable,value)
 else:result['status']='DEADLINE'
 result.update(seconds=time.monotonic()-start,counts=counts,oracle=oracle.stats,support_inserter=insert.stats,best_score=best);(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(output/'exact-pair-cache.json').write_text(json.dumps([dict(pair=list(k[:2]),values=list(k[2:]),area=v)for k,v in oracle.exact_areas.items()])+'\n');return result
