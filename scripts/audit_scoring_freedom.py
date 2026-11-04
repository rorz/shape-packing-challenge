"""Bounded audit of geometric equivalence, input ordering, and rounding seams."""
from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from semicircle_packing.exact_area import polygon_intersection_areas_and_jacobian
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle, canonicalize_theta, farthest_boundary_point_from, semicircle_boundary_points
from semicircle_packing.scoring import compute_mec, minimum_enclosing_circle, validate_and_score


def load(path):
    data = json.loads(path.read_text())
    return np.array([[round(float(q["x"]), 6), round(float(q["y"]), 6),
                      canonicalize_theta(q["theta"])] for q in data])


def placements(points):
    return [Semicircle(*q) for q in points]


def save(path, points):
    path.write_text(json.dumps([dict(x=float(x), y=float(y), theta=float(t)) for x,y,t in points], indent=2) + "\n")


def circle_record(points):
    shapes = placements(points)
    cx, cy, radius = compute_mec(shapes)
    excess = max(math.hypot(*(np.array(farthest_boundary_point_from(q,cx,cy))-[cx,cy]))-radius for q in shapes)
    return dict(center=[cx,cy], radius=radius, max_excess=float(excess), contained=bool(excess <= 1e-9))


def refinement_trace(points):
    shapes = placements(points)
    samples = np.vstack([semicircle_boundary_points(q,128) for q in shapes])
    history = []
    for iteration in range(21):
        cx,cy,radius = minimum_enclosing_circle(samples)
        far = np.array([farthest_boundary_point_from(q,cx,cy) for q in shapes])
        excess = np.linalg.norm(far-[cx,cy],axis=1)-radius
        mask = excess > 1e-12
        history.append(dict(iteration=iteration,points=len(samples),radius=radius,
                            max_excess=float(excess.max()),added=int(mask.sum())))
        if not mask.any() or iteration == 20:
            break
        samples = np.vstack([samples,far[mask]])
    return history


def high_precision_support_radius(points, indices):
    """Circumradius of three exact decimal disk centers, plus unit radius."""
    with localcontext() as context:
        context.prec = 70
        vertices = [[Decimal(str(float(v))) for v in points[i,:2]] for i in indices]
        def squared(a,b):return sum((a[k]-b[k])**2 for k in (0,1))
        a,b,c = vertices
        ab,bc,ca = squared(a,b),squared(b,c),squared(c,a)
        cross = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        radius = (ab*bc*ca).sqrt()/(2*abs(cross))+1
        return dict(indices=indices,radius=str(radius),acute=bool(max(ab,bc,ca)<ab+bc+ca-max(ab,bc,ca)))


def tie_patterns(values):
    choices = {}
    epsilons = np.linspace(-2e-15,2e-15,4097)
    for sign in (-1,1):
        for epsilon in epsilons:
            offset = sign*.5e-6 + float(epsilon)
            rounded = np.array([round(float(v+offset),6) for v in values])
            units = tuple(np.rint((rounded-values)*1e6).astype(int))
            choices.setdefault(units,dict(offset=offset,values=rounded))
    return list(choices.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("solution.json"))
    parser.add_argument("--output-dir",type=Path,default=Path("experiments/2026-11-04"))
    parser.add_argument("--seed",type=int,default=20261104)
    parser.add_argument("--orders",type=int,default=1024)
    parser.add_argument("--seconds",type=float,default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    p = load(args.input)
    baseline = circle_record(p)
    baseline_validation = validate_and_score(placements(p))
    report = dict(input=str(args.input),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  seed=args.seed, baseline=baseline,baseline_valid=baseline_validation.valid,
                  fast_mec=compute_mec_fast(p),refinement=refinement_trace(p),
                  exact_support=high_precision_support_radius(p,[3,5,7]))
    (args.output_dir/"scoring-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    best_score = baseline["radius"]
    best_points = p.copy()
    rng = np.random.default_rng(args.seed)
    order_results = []
    best_order = np.arange(15)
    orders = [np.roll(np.arange(15),i) for i in range(15)] + [np.arange(14,-1,-1)]
    orders += [rng.permutation(15) for _ in range(args.orders)]
    for index,order in enumerate(orders):
        if time.monotonic()-start > args.seconds:break
        value = circle_record(p[order])
        order_results.append(dict(order=order.tolist(),**value))
        if value["contained"] and value["radius"] < best_score:
            best_score,best_points,best_order = value["radius"],p[order].copy(),order
    report["orders"] = dict(count=len(order_results),minimum=min(v["radius"]for v in order_results),
                            maximum=max(v["radius"]for v in order_results),
                            invalid=sum(not v["contained"]for v in order_results),results=order_results)
    report["best_order_refinement"] = refinement_trace(p[best_order])
    print(json.dumps(dict(stage="orders",**{k:v for k,v in report["orders"].items()if k!="results"},seconds=time.monotonic()-start)),flush=True)
    translations = []
    for magnitude in (1,10,100,1000,10000):
        for dx,dy in ((magnitude,0),(0,magnitude),(magnitude,magnitude),(-magnitude,magnitude)):
            if time.monotonic()-start > args.seconds:break
            candidate = p.copy()
            candidate[:,:2] = [[round(float(x+dx),6),round(float(y+dy),6)] for x,y in p[:,:2]]
            value = circle_record(candidate)
            centered = candidate.copy();centered[:,:2]-=[dx,dy]
            value.update(translation=[dx,dy],roundoff=float(np.abs(centered[:,:2]-p[:,:2]).max()),
                         stable_radius=compute_mec_fast(centered)[2],
                         within_server_bounds=bool(np.abs(candidate[:,:2]).max() <= 100))
            translations.append(value)
            if value["within_server_bounds"] and value["contained"] and value["radius"] < best_score:
                validation = validate_and_score(placements(candidate))
                if validation.valid:
                    best_score,best_points = validation.score,candidate.copy()
    report["translations"] = translations
    print(json.dumps(dict(stage="translations",minimum=min(v["radius"]for v in translations),seconds=time.monotonic()-start)),flush=True)
    isometries = []
    for swap,sx,sy in itertools.product((False,True),(-1,1),(-1,1)):
        matrix = np.array([[0,sx],[sy,0]]) if swap else np.diag([sx,sy])
        candidate = p.copy();candidate[:,:2]=p[:,:2]@matrix.T
        normals=np.column_stack([np.cos(p[:,2]),np.sin(p[:,2])])@matrix.T
        candidate[:,2]=[canonicalize_theta(t)for t in np.arctan2(normals[:,1],normals[:,0])]
        validation=validate_and_score(placements(candidate))
        isometries.append(dict(swap=swap,sx=sx,sy=sy,valid=validation.valid,score=validation.score,errors=validation.errors))
        if validation.valid and validation.score < best_score:
            best_score,best_points=validation.score,candidate.copy()
    report["grid_isometries"] = isometries
    x_patterns,y_patterns=tie_patterns(p[:,0]),tie_patterns(p[:,1])
    tie_results=[];prefiltered=0
    for x,y in itertools.product(x_patterns,y_patterns):
        if time.monotonic()-start > args.seconds:break
        candidate=p.copy();candidate[:,0]=x["values"];candidate[:,1]=y["values"]
        areas,_=polygon_intersection_areas_and_jacobian(candidate)
        if areas.max()>1e-6+2e-11:continue
        prefiltered+=1
        validation=validate_and_score(placements(candidate))
        tie_results.append(dict(offset=[x["offset"],y["offset"]],valid=validation.valid,
                                score=validation.score,errors=validation.errors))
        if validation.valid and validation.score < best_score:
            best_score,best_points=validation.score,candidate.copy()
    report["half_grid_ties"] = dict(x_patterns=len(x_patterns),y_patterns=len(y_patterns),
                                   combinations=len(x_patterns)*len(y_patterns),prefiltered=prefiltered,results=tie_results)
    report["best_score"] = best_score
    report["improvement"] = baseline["radius"]-best_score
    report["seconds"] = time.monotonic()-start
    save(args.output_dir/"scoring-best.json",best_points)
    (args.output_dir/"scoring-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items()if k in("best_score","improvement","seconds")}),flush=True)


if __name__=="__main__":main()
