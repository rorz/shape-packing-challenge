"""Amortized exact-overlap benchmark using opposite-domain outer preclipping.

For every A in a finite domain, A is contained in outerA; similarly B is in
outerB. Thus (A intersect outerB) intersect (B intersect outerA) = A intersect B.
The outer polygons use conservative continuous-semicircle support bounds, so
contain the scorer's inscribed4096-point polygons. GEOS remains authoritative.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from semicircle_packing.geometry import Semicircle,semicircle_polygon


def outer_polygon(domain,uniform_directions=128,pad=1e-10):
    """Intersect support halfplanes covering every continuous source pose."""
    domain=np.asarray(domain,float)
    if domain.ndim!=2 or domain.shape[1]!=3 or not len(domain)or not np.isfinite(domain).all()or pad<=0:
        raise ValueError('Expected nonempty finite Nx3 domain and positive pad')
    theta=np.r_[np.arange(uniform_directions)*2*np.pi/uniform_directions,domain[:,2]+np.pi]
    theta=np.unique(np.mod(theta,2*np.pi));normals=np.c_[np.cos(theta),np.sin(theta)]
    facing=np.c_[np.cos(domain[:,2]),np.sin(domain[:,2])]
    tangent=np.c_[-facing[:,1],facing[:,0]]
    support=domain[:,:2]@normals.T+np.where(facing@normals.T>=0,1.,np.abs(tangent@normals.T))
    bounds=support.max(axis=0)+pad
    lo=domain[:,:2].min(axis=0)-1-pad;hi=domain[:,:2].max(axis=0)+1+pad
    vertices=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    for normal,bound in zip(normals,bounds):
        previous=np.roll(vertices,1,axis=0);d1=previous@normal-bound;d2=vertices@normal-bound
        inside=d2<=0;cross=(d1<=0)!=inside
        points=[]
        for start,end,a,b,keep,intersects in zip(previous,vertices,d1,d2,inside,cross):
            if intersects:points.append(start+(end-start)*(a/(a-b)))
            if keep:points.append(end)
        vertices=np.asarray(points)
        if not len(vertices):raise ValueError('Empty outer support intersection')
    # Roundoff can leave tiny inward dents between nearly coincident clipping
    # planes. Convex hull only enlarges the outer bound and restores convexity.
    polygon=Polygon(vertices).convex_hull
    if not polygon.is_valid:raise ValueError('Invalid outer support polygon')
    def containment_margin(poly):
        points=np.asarray(poly.exterior.coords)[:-1];edges=np.roll(points,-1,axis=0)-points
        lengths=np.linalg.norm(edges,axis=1);keep=lengths>0;points=points[keep];edges=edges[keep];lengths=lengths[keep]
        directions=np.c_[edges[:,1],-edges[:,0]]/lengths[:,None]
        if not poly.exterior.is_ccw:directions*=-1
        actual_bounds=np.sum(points*directions,axis=1)
        actual_support=domain[:,:2]@directions.T+np.where(facing@directions.T>=0,1.,np.abs(tangent@directions.T))
        return float(np.min(actual_bounds-actual_support.max(axis=0)))
    margin=containment_margin(polygon);box_fallback=margin<pad*.1
    if box_fallback:
        # Preserve inclusion for unusual numerical/coordinate scales. A box
        # still bounds every unit disk, though it may give less acceleration.
        polygon=Polygon([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
        margin=containment_margin(polygon)
        if margin<=0:raise ValueError('Cannot certify outer containment at this coordinate scale')
    return polygon,dict(directions=len(normals),vertices=len(polygon.exterior.coords)-1,pad=pad,support_maximum_excess=float(np.max(support-bounds[None,:])),minimum_continuous_margin=margin,box_fallback=box_fallback)


def area_matrix(a,b,chunk=1024):
    result=np.empty((len(a),len(b)));ii,jj=np.indices(result.shape);ii=ii.ravel();jj=jj.ravel()
    for begin in range(0,len(ii),chunk):
        x=ii[begin:begin+chunk];y=jj[begin:begin+chunk]
        result[x,y]=shapely.area(shapely.intersection(a[x],b[y]))
    return result


def clipped_area_matrix(full_a,full_b,clipped_a,clipped_b,threshold_guard=1e-12):
    result=area_matrix(clipped_a,clipped_b)
    near=np.abs(result-1e-6)<=threshold_guard;ii,jj=np.nonzero(near)
    if len(ii):result[ii,jj]=shapely.area(shapely.intersection(full_a[ii],full_b[jj]))
    return result,int(near.sum())


def benchmark(a,b,index_a,index_b,*,uniform_directions=128,pad=1e-10):
    start=time.perf_counter();full_a=np.asarray([semicircle_polygon(Semicircle(*q))for q in a[index_a]],dtype=object);full_b=np.asarray([semicircle_polygon(Semicircle(*q))for q in b[index_b]],dtype=object);polygon_seconds=time.perf_counter()-start
    start=time.perf_counter();outer_a,meta_a=outer_polygon(a,uniform_directions,pad);outer_b,meta_b=outer_polygon(b,uniform_directions,pad)
    clipped_a=shapely.intersection(full_a,outer_b);clipped_b=shapely.intersection(full_b,outer_a);setup_seconds=time.perf_counter()-start
    # Containment audit uses every sampled full polygon. Analytic support
    # construction above covers the complete finite domains, not only samples.
    start=time.perf_counter();missing_a=shapely.area(shapely.difference(full_a,outer_a));missing_b=shapely.area(shapely.difference(full_b,outer_b));containment_seconds=time.perf_counter()-start
    start=time.perf_counter();reference=area_matrix(full_a,full_b);full_seconds=time.perf_counter()-start
    start=time.perf_counter();raw_clipped=area_matrix(clipped_a,clipped_b);clipped_seconds=time.perf_counter()-start
    start=time.perf_counter();corrected,fallbacks=clipped_area_matrix(full_a,full_b,clipped_a,clipped_b);corrected_seconds=time.perf_counter()-start
    error=np.abs(reference-raw_clipped);wrong_raw=(reference>1e-6)!=(raw_clipped>1e-6);wrong_corrected=(reference>1e-6)!=(corrected>1e-6)
    coords=lambda values:dict(minimum=int(shapely.get_num_coordinates(values).min()),median=float(np.median(shapely.get_num_coordinates(values))),maximum=int(shapely.get_num_coordinates(values).max()))
    report=dict(domain_sizes=[len(a),len(b)],sample_sizes=[len(index_a),len(index_b)],entries=reference.size,outer_a=meta_a,outer_b=meta_b,full_coordinates_a=coords(full_a),full_coordinates_b=coords(full_b),clipped_coordinates_a=coords(clipped_a),clipped_coordinates_b=coords(clipped_b),polygon_seconds=polygon_seconds,preclip_seconds=setup_seconds,containment_audit_seconds=containment_seconds,full_geos_seconds=full_seconds,clipped_geos_seconds=clipped_seconds,clipped_with_fallback_seconds=corrected_seconds,maximum_missing_outer_area=float(max(missing_a.max(),missing_b.max())),maximum_area_error=float(error.max()),mean_area_error=float(error.mean()),classification_mismatches_raw=int(wrong_raw.sum()),classification_mismatches_after_fallback=int(wrong_corrected.sum()),near_threshold_full_fallbacks=fallbacks,intersection_speedup=full_seconds/corrected_seconds,speedup_including_preclip=full_seconds/(setup_seconds+corrected_seconds))
    return report,dict(indices_a=index_a,indices_b=index_b,reference=reference,clipped=raw_clipped,corrected=corrected,outer_a=np.asarray(outer_a.exterior.coords),outer_b=np.asarray(outer_b.exterior.coords))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--domains',type=Path,default=Path('experiments/2026-11-04/round3/conflicts/phase-pool-search.domains.npz'));ap.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/round3/mirror/preclip'));ap.add_argument('--samples',type=int,default=128);ap.add_argument('--directions',type=int,default=128);ap.add_argument('--pad',type=float,default=1e-10);ap.add_argument('--seed',type=int,default=804173300);ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--pairs',default='0,1;0,2;4,8;5,8');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    snapshot=a.output/'source-domains.npz';shutil.copy2(a.domains,snapshot);digest=hashlib.sha256(snapshot.read_bytes()).hexdigest();domains=np.load(snapshot);rng=np.random.default_rng(a.seed);start=time.monotonic();reports=[]
    (a.output/'input.json').write_text(json.dumps(vars(a)|dict(domains=str(a.domains),output=str(a.output),source_sha256=digest,shapely_version=shapely.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    for text in a.pairs.split(';'):
        if time.monotonic()-start>=a.seconds:break
        i,j=map(int,text.split(','));da=domains[f'p{i}'];db=domains[f'p{j}'];ia=np.sort(rng.choice(len(da),min(a.samples,len(da)),replace=False));ib=np.sort(rng.choice(len(db),min(a.samples,len(db)),replace=False))
        report,state=benchmark(da,db,ia,ib,uniform_directions=a.directions,pad=a.pad);report.update(pair=[i,j],seconds_total=time.monotonic()-start);reports.append(report);np.savez_compressed(a.output/f'pair-{i}-{j}.npz',**state);(a.output/f'pair-{i}-{j}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if report['maximum_missing_outer_area']>1e-18 or report['classification_mismatches_after_fallback']:raise AssertionError('Preclip benchmark failed containment/classification audit')
    (a.output/'report.json').write_text(json.dumps(dict(pairs=reports,source_sha256=digest,seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
