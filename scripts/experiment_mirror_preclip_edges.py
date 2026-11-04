"""Separate threshold and continuous outer-containment controls for preclipping."""
from pathlib import Path
import json,time
import numpy as np
import shapely
from scipy.optimize import brentq
from semicircle_packing.geometry import Semicircle,semicircle_polygon
from semicircle_packing.exact_area import polygon_intersection_areas_and_jacobian
from experiment_mirror_preclip import outer_polygon,benchmark


def outer_edge_margin(domain,outer):
    xy=np.asarray(outer.exterior.coords)[:-1];edge=np.roll(xy,-1,axis=0)-xy;length=np.linalg.norm(edge,axis=1);keep=length>1e-13;xy=xy[keep];edge=edge[keep];length=length[keep]
    normals=np.c_[edge[:,1],-edge[:,0]]/length[:,None]
    if not outer.exterior.is_ccw:normals*=-1
    bounds=np.sum(xy*normals,axis=1)
    facing=np.c_[np.cos(domain[:,2]),np.sin(domain[:,2])];tangent=np.c_[-facing[:,1],facing[:,0]]
    support=domain[:,:2]@normals.T+np.where(facing@normals.T>=0,1.,np.abs(tangent@normals.T))
    return dict(minimum_continuous_margin=float((bounds-support.max(axis=0)).min()),minimum_edge_length=float(length.min()),edges=len(length),poses=len(domain))


def main():
    root=Path('experiments/2026-11-04/round3/mirror/preclip');domains=np.load(root/'source-domains.npz');reports=[];containment=[];start=time.monotonic()
    for i in [0,1,2,4,5,8]:
        domain=domains[f'p{i}'];outer,_=outer_polygon(domain);audit=outer_edge_margin(domain,outer);audit['piece']=i;containment.append(audit)
        if audit['minimum_continuous_margin'] < -1e-13:raise AssertionError('Continuous domain escapes computed outer edge')
    for i,j in [(0,1),(4,8),(5,8)]:
        a=domains[f'p{i}'][0];b=domains[f'p{j}'][0];_,jac=polygon_intersection_areas_and_jacobian(np.vstack([a,b]));direction=jac[0,3:5].copy();direction/=np.linalg.norm(direction)
        pa=semicircle_polygon(Semicircle(*a))
        def area(step):
            q=b.copy();q[:2]+=step*direction
            return pa.intersection(semicircle_polygon(Semicircle(*q))).area
        choices=[];requested=[]
        for delta in [0.,-1e-13,1e-13,-5e-13,5e-13,-2e-12,2e-12,-1e-11,1e-11]:
            target=1e-6+delta;lo=-.002;hi=.002
            if(area(lo)-target)*(area(hi)-target)>0:raise AssertionError('Threshold root not bracketed')
            step=brentq(lambda x:area(x)-target,lo,hi,xtol=1e-15,rtol=1e-15);q=b.copy();q[:2]+=step*direction;choices.append(q);requested.append(target)
        choices=np.array(choices);report,state=benchmark(a[None,:],choices,np.array([0]),np.arange(len(choices)));report.update(pair=[i,j],requested_area=requested,actual_reference_area=state['reference'].ravel().tolist());reports.append(report);np.savez_compressed(root/f'threshold-{i}-{j}.npz',poses_a=a[None,:],poses_b=choices,**state)
        if report['classification_mismatches_after_fallback'] or report['near_threshold_full_fallbacks']<5:raise AssertionError('Threshold fallback audit failed')
    cases=[('disjoint',np.array([[0.,0.,0.]]),np.array([[4.,0.,np.pi]])),('identical',np.array([[0.,0.,0.]]),np.array([[0.,0.,0.]])),('opposed_touch',np.array([[0.,0.,0.]]),np.array([[0.,0.,np.pi]])),('seam_alias_geometry',np.array([[0.,0.,3.141593],[0.,0.,-3.141593]]),np.array([[-1.99985,0.,0.]]))]
    controls=[]
    for name,a,b in cases:
        report,_=benchmark(a,b,np.arange(len(a)),np.arange(len(b)));report['case']=name;controls.append(report)
        if report['classification_mismatches_after_fallback']or report['maximum_missing_outer_area']>1e-18:raise AssertionError(name)
    output=dict(continuous_outer_edge_audits=containment,threshold_cases=reports,additional_controls=controls,seconds=time.monotonic()-start);(root/'edge-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(continuous_outer_edge_audits=containment,threshold_fallback_counts=[r['near_threshold_full_fallbacks']for r in reports],threshold_maximum_area_errors=[r['maximum_area_error']for r in reports],controls=[r['case']for r in controls],seconds=output['seconds']),indent=2))

if __name__=='__main__':main()
