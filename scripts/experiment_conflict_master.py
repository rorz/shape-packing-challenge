"""Finite exact-polygon placement CSP with free-center geometric conflict cuts.

Each categorical choice contains all three placement coordinates. Domains are
recombinations of complete LLL moves, including their correlated angle changes.
No continuous MEC center is frozen. A rejected enclosure learns a no-good on
at most three shapes whose physical boundary points already need too large a
circle. INFEASIBLE is only a statement about the explicitly saved finite domains.
"""
from __future__ import annotations

import os
for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import shapely
from ortools.sat.python import cp_model
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle, canonicalize_theta, farthest_boundary_point_from, semicircle_polygon
from semicircle_packing.polish import checked, load, save
from semicircle_packing.fast_geometry import FastGeometry


def correlated_domains(base, moves=None, *, directions=4, jitter=2, pair_sums=False, local_steps=0,
                       minkowski_steps=0,max_domain=4096):
    """Produce finite domains and retain the selected integer directions."""
    base = np.round(np.asarray(base, dtype=float), 6)
    states = [base]
    indices = []
    if moves is not None and directions:
        moves = np.rint(np.asarray(moves, dtype=float)).copy()
        # Remove common integer translation. Keep every angular displacement.
        moves[:, :, :2] -= moves[:, :1, :2]
        norms = np.linalg.norm(moves[:, :, :2].reshape(len(moves), -1), axis=1)
        indices = [int(k) for k in np.argsort(norms) if norms[k] > 0][:directions]
        selected = moves[indices]
        offsets = [sign * move for move in selected for sign in (-1, 1)]
        if pair_sums:
            offsets += [a * selected[i] + b * selected[j]
                        for i in range(len(selected)) for j in range(i + 1, len(selected))
                        for a, b in itertools.product((-1, 1), repeat=2)]
        states += [base + offset * 1e-6 for offset in offsets]
    domains = []
    for piece in range(len(base)):
        values = []
        seen = set()
        # Base first lets the objective and hint recover the positive control.
        for state in states:
            angular_width=max(jitter,minkowski_steps)
            xy=[(0,0),*(v for v in itertools.product(range(-minkowski_steps,minkowski_steps+1),repeat=2)if v!=(0,0))]
            for angular in [0, *range(-angular_width, 0), *range(1, angular_width + 1)]:
                for x,y in xy:
                    value = np.round(state[piece]+np.asarray([x,y,angular])*1e-6,6)
                    value[2] = canonicalize_theta(value[2])
                    if canonicalize_theta(value[2]) != value[2]:
                        raise ValueError('Domain crosses an unstable arena angle alias; store raw and canonical angles separately')
                    key = tuple(np.rint(value * 1e6).astype(np.int64))
                    if key not in seen:
                        seen.add(key)
                        values.append(value)
        if local_steps:
            for delta in itertools.product(range(-local_steps,local_steps+1), repeat=3):
                value = np.round(base[piece] + np.asarray(delta)*1e-6, 6)
                value[2] = canonicalize_theta(value[2])
                if canonicalize_theta(value[2]) != value[2]:
                    raise ValueError('Local domain crosses an unstable arena angle alias')
                key = tuple(np.rint(value*1e6).astype(np.int64))
                if key not in seen:
                    seen.add(key)
                    values.append(value)
        if len(values)>max_domain:
            raise ValueError(f'Piece {piece} has {len(values)} choices above explicit cap {max_domain}; no truncation performed')
        domains.append(np.asarray(values))
    return domains, indices


POLYGON_LOSS = np.pi-4095*np.sin(np.pi/4095)  # Combined deficit of two 4096-vertex semicircles.


def separated_domains(a,b):
    separation=np.maximum(np.maximum(a[:,:2].min(0)-b[:,:2].max(0),b[:,:2].min(0)-a[:,:2].max(0)),0)
    return np.linalg.norm(separation)>2


def exact_tables(domains, deadline=float('inf'), *, band_screen=False, arithmetic_guard=1e-9,stats=None,cache_dir=None,local_loss=False):
    """Build the scorer's 4096-point GEOS compatibility tables, without Taylor terms."""
    if arithmetic_guard<1e-9:
        raise ValueError('Area band arithmetic allowance must be at least 1e-9')
    polygons = [[None]*len(d)for d in domains]
    if stats is None:stats={}
    stats.update(native_pairs=0,screen_allowed=0,screen_rejected=0,geos_pairs=0,cache_pairs=0,
                 polygon_loss_bound=float(POLYGON_LOSS),arithmetic_guard=arithmetic_guard,local_loss=local_loss)
    authority=hashlib.sha256(Path(__file__).read_bytes()+
        (Path(__file__).parents[1]/'src/semicircle_packing/area_kernel.cpp').read_bytes()+
        (Path(__file__).parents[1]/'src/semicircle_packing/geometry.py').read_bytes()+
        (Path(__file__).parents[1]/'src/semicircle_packing/config.py').read_bytes()+
        Path(__file__).with_name('_conflict_area_band.cpp').read_bytes()+
        Path(__file__).with_name('conflict_area_band.py').read_bytes()+
        str(shapely.__version__).encode()).hexdigest()
    if cache_dir:
        cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    started=last_notice=time.monotonic()
    if band_screen:
        from experiment_contact_csp import pair_kernel
        kernel=pair_kernel()
    def polygon(piece,index):
        if polygons[piece][index] is None:
            polygons[piece][index]=semicircle_polygon(Semicircle(*domains[piece][index]))
        return polygons[piece][index]
    tables = []
    count = 0
    for i, j in itertools.combinations(range(len(domains)), 2):
        if time.monotonic() >= deadline:
            raise TimeoutError('Table generation reached deadline')
        a, b = domains[i], domains[j]
        # Unit disk separation is a safe pair omission, irrespective of angles.
        if separated_domains(a,b):
            continue
        cache_path=None
        if cache_dir:
            digest=hashlib.sha256(a.tobytes()+b.tobytes()+str((band_screen,arithmetic_guard,local_loss,authority)).encode()).hexdigest()[:20]
            cache_path=cache_dir/f'{i}_{j}_{digest}.npz'
            if cache_path.exists():
                packed=np.load(cache_path)
                mask=np.unpackbits(packed['mask'],count=len(a)*len(b)).reshape(len(a),len(b)).astype(bool)
                if not mask.all():tables.append((i,j,np.column_stack(np.nonzero(mask)).astype(int)))
                stats['cache_pairs']+=1
                continue
        if band_screen:
            if local_loss:
                from conflict_area_band import true_area_loss_band
                true_area,loss_bound=true_area_loss_band(a,b)
            else:
                true_area=np.empty((len(a),len(b)))
                kernel(len(a),len(b),np.ascontiguousarray(a),np.ascontiguousarray(b),true_area)
                loss_bound=POLYGON_LOSS
            mask=true_area<=1e-6-arithmetic_guard
            rejected=true_area>1e-6+loss_bound+arithmetic_guard
            uncertain=~(mask|rejected)
            stats['native_pairs']+=true_area.size
            stats['screen_allowed']+=int(mask.sum())
            stats['screen_rejected']+=int(rejected.sum())
        else:
            mask=np.zeros((len(a),len(b)),dtype=bool)
            uncertain=np.ones_like(mask)
        ia,jb=np.nonzero(uncertain)
        for begin in range(0,len(ia),1024):
            if time.monotonic()>=deadline:raise TimeoutError('Exact band completion reached deadline')
            ii,jj=ia[begin:begin+1024],jb[begin:begin+1024]
            pa=np.asarray([polygon(i,int(k))for k in ii],dtype=object)
            pb=np.asarray([polygon(j,int(k))for k in jj],dtype=object)
            areas=shapely.area(shapely.intersection(pa,pb))
            mask[ii,jj]=areas<=1e-6
            count+=len(ii)
        stats['geos_pairs']=count
        if not mask.all():
            allowed = np.column_stack(np.nonzero(mask)).astype(int)
            tables.append((i, j, allowed))
        if cache_path:
            np.savez_compressed(cache_path,mask=np.packbits(mask),shape=np.asarray(mask.shape))
        if time.monotonic()-last_notice>10:
            print('PAIR_PROGRESS',i,j,'native',stats['native_pairs'],'GEOS',count,'elapsed',time.monotonic()-started,flush=True)
            last_notice=time.monotonic()
    return tables, count


def compress_pair_relation(mask):
    """Quotient only literally identical rows and columns; retain every tuple."""
    unique_rows,row_map=np.unique(np.packbits(mask,axis=1),axis=0,return_inverse=True)
    rows=np.unpackbits(unique_rows,count=mask.shape[1],axis=1).astype(bool)
    _,column_indices,column_map=np.unique(np.packbits(rows.T,axis=1),axis=0,
                                         return_index=True,return_inverse=True)
    return row_map,column_map,rows[:,column_indices]


def add_pair_constraint(model,first,second,allowed,ni,nj):
    """Encode exact row/column classes, then the smaller compatibility relation."""
    mask=np.zeros((ni,nj),dtype=bool)
    mask[allowed[:,0],allowed[:,1]]=True
    row_map,column_map,small=compress_pair_relation(mask)
    if small.shape[0]<ni:
        row=model.new_int_var(0,small.shape[0]-1,f'row_{first.index}_{second.index}')
        model.add_element(first,row_map.tolist(),row)
    else:row=first
    if small.shape[1]<nj:
        column=model.new_int_var(0,small.shape[1]-1,f'column_{first.index}_{second.index}')
        model.add_element(second,column_map.tolist(),column)
    else:column=second
    # Even when no quotient exists, np.unique may permute class indices.
    if small.shape[0]==ni:
        small=small[row_map]
    if small.shape[1]==nj:
        small=small[:,column_map]
    if small.sum()>small.size//2:
        tuples=np.column_stack(np.nonzero(~small)).tolist()
        model.add_forbidden_assignments([row,column],tuples)
    else:
        tuples=np.column_stack(np.nonzero(small)).tolist()
        model.add_allowed_assignments([row,column],tuples)
    return [ni,nj,*small.shape]


def boundary_conflict(placements, center, target, guard=2e-10):
    """Return <=3 shape IDs whose actual points require radius > target+guard.

    A triangle's minimum circle is either its longest-side diameter or its
    circumcircle when acute. Maximizing these point-triple radii supplies a
    lower bound independent of the unassigned shapes and enclosing center.
    The guard is numerical, not an interval-arithmetic proof certificate.
    """
    points, owners = [], []
    for i, p in enumerate(placements):
        normal = np.array([-np.sin(p[2]), np.cos(p[2])])
        points.extend((p[:2] + normal, p[:2] - normal,
                       np.asarray(farthest_boundary_point_from(Semicircle(*p), *center))))
        owners.extend((i, i, i))
    points, owners = np.asarray(points), np.asarray(owners)
    triples = np.asarray(list(itertools.combinations(range(len(points)), 3)), dtype=int)
    a, b, c = (points[triples[:, k]] for k in range(3))
    ab, bc, ca = b-a, c-b, a-c
    aa, bb, cc = (np.sum(v*v, axis=1) for v in (ab, bc, ca))
    longest = np.maximum.reduce((aa, bb, cc))
    radii = .5 * np.sqrt(longest)
    acute = 2 * longest < aa + bb + cc
    cross = np.abs(ab[:, 0] * (c-a)[:, 1] - ab[:, 1] * (c-a)[:, 0])
    good = acute & (cross > 1e-14)
    radii[good] = np.sqrt(aa[good]*bb[good]*cc[good]) / (2*cross[good])
    best = int(np.argmax(radii))
    if radii[best] > target + guard:
        return sorted(set(map(int, owners[triples[best]]))), float(radii[best])
    return None, float(radii[best])


def overlap_conflict_core(domains, tables, seconds=3):
    """Name a solver-proved conflicting subset of finite pair tables."""
    model = cp_model.CpModel()
    variables = [model.new_int_var(0, len(d)-1, f'p{i}') for i,d in enumerate(domains)]
    assumptions = []
    for k, (i,j,allowed) in enumerate(tables):
        enabled = model.new_bool_var(f'pair_{i}_{j}')
        constraint=model.add_allowed_assignments([variables[i],variables[j]],allowed.tolist())
        constraint.only_enforce_if(enabled)
        assumptions.append(enabled)
    model.add_assumptions(assumptions)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status != cp_model.INFEASIBLE:
        return dict(status=solver.status_name(status))
    indices = set(solver.sufficient_assumptions_for_infeasibility())
    core = [k for k,a in enumerate(assumptions) if a.index in indices]
    return dict(status='INFEASIBLE', pairs=[[tables[k][0],tables[k][1]] for k in core],
                note='A sufficient core, not necessarily minimum; finite domains only')


def categorical_support_cut(domains, candidate, circle, scale=10**12):
    """Additive fixed-direction lower bound valid for every domain assignment.

    Nonnegative weights sum to one and their outward directions balance. Shape
    support is exact, including endpoint switches. Numerical imbalance is
    covered by a conservative bound on a feasible enclosing center's norm.
    Integer coefficients round down; the offset additionally covers arithmetic.
    """
    from experiment_hillclimb import dual_weights
    center = np.asarray(circle[:2])
    active, weights = dual_weights(candidate, center, circle[2])
    if not weights.shape[1]:
        return None
    relative = candidate.copy()
    relative[:, :2] -= center
    _, jac = FastGeometry(len(candidate)).boundary(np.r_[relative.ravel(),circle[2]])
    directions = np.asarray([[-jac[k,3*(k//3)],-jac[k,3*(k//3)+1]] for k in active])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    pieces = active//3
    def support(p,direction):
        co,sn = np.cos(p[:,2]),np.sin(p[:,2])
        curved = co*direction[0]+sn*direction[1]>=0
        unit = np.where(curved,1.,np.abs(-sn*direction[0]+co*direction[1]))
        return (p[:,:2]-center)@direction+unit
    values = np.asarray([support(candidate[i:i+1],u)[0] for i,u in zip(pieces,directions)])
    # If an assignment has an enclosing circle R<=target, its center is within
    # max domain-center norm +1+target. The extra radius below is conservative
    # because all target radii in this experiment are <=10.
    center_bound = max(np.linalg.norm(d[:,:2]-center,axis=1).max() for d in domains)+11.
    imbalance = np.linalg.norm(directions.T@weights,axis=0)
    bounds = values@weights-center_bound*imbalance
    chosen = int(np.argmax(bounds))
    ww = weights[:,chosen]
    contributions = {}
    for i,u,w in zip(pieces,directions,ww):
        if w<=0:
            continue
        i=int(i)
        contributions[i] = contributions.get(i,np.zeros(len(domains[i])))+w*support(domains[i],u)
    coefficients = {i:np.floor(v*scale).astype(np.int64) for i,v in contributions.items()}
    offset = int(np.ceil((center_bound*imbalance[chosen]+1e-11)*scale))
    return dict(coefficients=coefficients, offset=offset, scale=scale,
                lower_bound=float(bounds[chosen]), residual=float(imbalance[chosen]),
                anchor=center.tolist(),center_bound=float(center_bound),
                direction_pieces=pieces.tolist(),directions=directions.tolist(),
                weights=ww.tolist())


def add_support_cut(model, variables, radius_variable, cut, serial):
    terms=[]
    for i,values in cut['coefficients'].items():
        term=model.new_int_var(int(values.min()),int(values.max()),f'support_{serial}_{i}')
        model.add_element(variables[i],values.tolist(),term)
        terms.append(term)
    model.add(sum(terms)<=radius_variable+cut['offset'])


def cut_record(cut):
    result={k:v for k,v in cut.items()if k!='coefficients'}
    result['coefficients']={str(k):v.tolist()for k,v in cut['coefficients'].items()}
    return result


def positive_control(domains,tables,expected_score=None):
    """Force the included incumbent, using the actual full tables and support cut."""
    base=np.asarray([d[0]for d in domains])
    validation=checked(base)
    if not validation.valid:
        raise AssertionError(f'Included base is not reference-valid: {validation.errors}')
    if expected_score is not None and abs(validation.score-expected_score)>1e-12:
        raise AssertionError(f'Unexpected control score {validation.score} != {expected_score}')
    target=validation.score+1e-10
    model=cp_model.CpModel()
    variables=[model.new_int_var(0,len(d)-1,f'control_{i}')for i,d in enumerate(domains)]
    for variable in variables:
        model.add(variable==0)
    for i,j,allowed in tables:
        add_pair_constraint(model,variables[i],variables[j],allowed,len(domains[i]),len(domains[j]))
    radius=model.new_int_var(0,int(np.floor(target*10**12)),'control_radius')
    cut=categorical_support_cut(domains,base,validation.mec)
    if cut is not None:
        add_support_cut(model,variables,radius,cut,'control')
    model.minimize(radius)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=10
    solver.parameters.num_search_workers=1
    status=solver.solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise AssertionError('Known valid included assignment rejected by actual tables/support constraints: '+solver.status_name(status))
    return dict(valid=True,score=validation.score,target=target,status=solver.status_name(status),
                forced_domain_indices=[0]*len(domains),support_lower_bound=cut['lower_bound']if cut else None)


def solve(domains, target, output, *, seconds=120, seed=110443, tables=None, support_cuts=False,
          check_positive=False,expected_score=None,band_screen=False,local_loss=False):
    start = time.monotonic()
    deadline = start + seconds
    np.savez_compressed(Path(output).with_suffix('.domains.npz'), **{f'p{i}': d for i,d in enumerate(domains)})
    if tables is None:
        table_stats={}
        try:
            tables,evaluations=exact_tables(domains,deadline,band_screen=band_screen,stats=table_stats,
                                            cache_dir=Path(output).with_suffix('.pair-cache'),local_loss=local_loss)
        except TimeoutError as error:
            report=dict(valid=False,incomplete_tables=True,reason=str(error),table_stats=table_stats,
                        seconds=time.monotonic()-start,scope='No feasibility conclusion; incomplete exact pair-table construction')
            Path(output).with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('DONE',json.dumps(report),flush=True)
            return report
    else:
        evaluations = 0
        table_stats={'resumed':True}
    model = cp_model.CpModel()
    variables = [model.new_int_var(0, len(d)-1, f'piece_{i}') for i, d in enumerate(domains)]
    compression=[]
    for i, j, allowed in tables:
        sizes=add_pair_constraint(model,variables[i],variables[j],allowed,len(domains[i]),len(domains[j]))
        compression.append([i,j,*sizes])
    costs = []
    for i, d in enumerate(domains):
        # A soft enumeration order only; it never removes placements.
        cost = np.rint(np.sum(((d-d[0])*1e6)**2, axis=1)).astype(int)
        cv = model.new_int_var(0, int(cost.max()), f'cost_{i}')
        model.add_element(variables[i], cost.tolist(), cv)
        model.add_hint(variables[i], 0)
        costs.append(cv)
    model.minimize(sum(costs))
    report = dict(target=target, domains=[len(d) for d in domains], pair_tables=len(tables),
                  exact_pair_evaluations=evaluations, candidates=0, cuts={}, best_radius=None,
                  seed=seed, scope='Only saved finite full-placement domains; free enclosing center',table_stats=table_stats,
                  pair_compression=compression)
    if check_positive:
        report['positive_control']=positive_control(domains,tables,expected_score)
        print('POSITIVE_CONTROL',json.dumps(report['positive_control']),flush=True)
    learned_support=[]
    if support_cuts:
        if not 0<target<=10:
            raise ValueError('Support-cut center bound requires 0<target<=10')
        radius_variable=model.new_int_var(0,int(np.floor(target*10**12)),'radius_lower_bound')
        initial=np.asarray([d[0]for d in domains])
        cut=categorical_support_cut(domains,initial,compute_mec_fast(initial))
        if cut is not None:
            add_support_cut(model,variables,radius_variable,cut,0)
            learned_support.append(cut_record(cut))
            report['support_cuts']=1
        model.minimize(radius_variable)
    np.savez_compressed(Path(output).with_suffix('.domains.npz'), **{f'p{i}': d for i,d in enumerate(domains)})
    domain_digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
    np.savez_compressed(Path(output).with_suffix('.tables.npz'),_domain_digest=np.asarray(domain_digest),
                        **{f'{i}_{j}': a for i,j,a in tables})
    print('MODEL', json.dumps(report), 'table_seconds', time.monotonic()-start, flush=True)
    while time.monotonic() < deadline:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(60 if not report['candidates']else 20,deadline-time.monotonic())
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        status = solver.solve(model)
        report['last_status'] = solver.status_name(status)
        if status == cp_model.INFEASIBLE:
            report['finite_exhausted'] = True
            if not report['candidates']:
                report['overlap_conflict_core'] = overlap_conflict_core(domains,tables)
            break
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if time.monotonic() >= deadline:
                break
            # Adapt solver seed/order, not the feasible set.
            seed += 1
            continue
        assignment = [solver.value(v) for v in variables]
        candidate = np.asarray([d[k] for d,k in zip(domains, assignment)])
        circle = compute_mec_fast(candidate)
        report['candidates'] += 1
        if report['best_radius'] is None or circle[2] < report['best_radius']:
            report['best_radius'] = circle[2]
            save(Path(output).with_suffix('.nearest.json'), candidate)
            print('NEAREST', report['candidates'], circle[2], 'elapsed', time.monotonic()-start, flush=True)
        improvement_threshold = report.get('best_valid_score',2.948526962953499)
        if circle[2] < improvement_threshold-2e-12:
            result = checked(candidate)
            if result.valid and result.score < improvement_threshold:
                report['best_valid_score'] = result.score
                save(Path(output).with_suffix('.best-valid.json'),candidate)
                print('VALID_IMPROVEMENT',result.score,flush=True)
        if circle[2] <= target + 2e-12:
            result = checked(candidate)
            if result.valid and result.score <= target + 2e-12:
                save(output, candidate)
                report['valid'] = True
                report['score'] = result.score
                print('VALID', result.score, flush=True)
                break
            report['reference_rejections'] = report.get('reference_rejections', []) + [dict(valid=result.valid, score=result.score, errors=result.errors)]
        if support_cuts:
            cut=categorical_support_cut(domains,candidate,circle)
            if cut is not None:
                lhs=sum(int(values[assignment[i]])for i,values in cut['coefficients'].items())
                if lhs>int(np.floor(target*cut['scale']))+cut['offset']:
                    serial=report.get('support_cuts',0)
                    add_support_cut(model,variables,radius_variable,cut,serial)
                    learned_support.append(cut_record(cut))
                    report['support_cuts']=serial+1
                    if report['candidates']%10==0:
                        print('SUPPORT_CUTS',report['support_cuts'],'pieces',list(cut['coefficients']),'bound',cut['lower_bound'],'elapsed',time.monotonic()-start,flush=True)
                    continue
        ids, bound = boundary_conflict(candidate, circle[:2], target)
        if ids is None:
            ids = list(range(len(domains)))
        model.add_forbidden_assignments([variables[i] for i in ids], [[assignment[i] for i in ids]])
        report['cuts'][str(len(ids))] = report['cuts'].get(str(len(ids)), 0) + 1
        if report['candidates'] % 25 == 0:
            print('CUTS', report['candidates'], report['cuts'], 'elapsed', time.monotonic()-start, flush=True)
    report['seconds'] = time.monotonic()-start
    report['valid'] = report.get('valid', False)
    if learned_support:
        Path(output).with_suffix('.support-cuts.json').write_text(json.dumps(learned_support,indent=2)+'\n')
    Path(output).with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DONE', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input')
    parser.add_argument('--basis')
    parser.add_argument('--resume-domains')
    parser.add_argument('--resume-tables')
    parser.add_argument('--support-cuts',action='store_true')
    parser.add_argument('--positive-control',action='store_true')
    parser.add_argument('--expected-score',type=float)
    parser.add_argument('--output', required=True)
    parser.add_argument('--directions', type=int, default=4)
    parser.add_argument('--jitter', type=int, default=2)
    parser.add_argument('--pair-sums', action='store_true')
    parser.add_argument('--local-steps', type=int, default=0)
    parser.add_argument('--minkowski-steps',type=int,default=0)
    parser.add_argument('--max-domain',type=int,default=4096)
    parser.add_argument('--band-screen',action='store_true')
    parser.add_argument('--local-loss',action='store_true')
    parser.add_argument('--seconds', type=float, default=240)
    parser.add_argument('--target', type=float, default=2.9485269238)
    parser.add_argument('--seed', type=int, default=110443)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_domains:
        archive=np.load(args.resume_domains)
        domains=[archive[f'p{i}']for i in range(len(archive.files))]
        selected=[]
    else:
        if not args.input:
            parser.error('--input or --resume-domains is required')
        base = load(args.input)
        moves = np.load(args.basis)['moves'] if args.basis else None
        domains, selected = correlated_domains(base, moves, directions=args.directions,
                                               jitter=args.jitter, pair_sums=args.pair_sums,local_steps=args.local_steps,
                                               minkowski_steps=args.minkowski_steps,max_domain=args.max_domain)
    tables=None
    if args.resume_tables:
        archive=np.load(args.resume_tables)
        tables=[(*map(int,k.split('_')),archive[k])for k in archive.files if not k.startswith('_')]
        if '_domain_digest'in archive.files:
            digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest()
            if str(archive['_domain_digest'])!=digest:
                raise ValueError('Resumed tables do not belong to these placement domains')
    input_path=args.resume_domains or args.input
    metadata = dict(arguments=vars(args), selected_basis_indices=selected,
                    input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest())
    output.with_suffix('.input.json').write_text(json.dumps(metadata, indent=2)+'\n')
    solve(domains, args.target, output, seconds=args.seconds, seed=args.seed,
          tables=tables,support_cuts=args.support_cuts,check_positive=args.positive_control,
          expected_score=args.expected_score,band_screen=args.band_screen,local_loss=args.local_loss)


if __name__ == '__main__':
    main()
