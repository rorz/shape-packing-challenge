"""Polish a successful grid phase while preserving its angular contact branch."""
from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, load, polygon_areas


def phase_polish(placements, allowance=1e-6, maxiter=400):
    """Fix piece zero's pose and the two opposed-flat relative orientations."""
    p = np.asarray(placements, dtype=float)
    constraint = AreaConstraints(p, allowance=allowance, fast_polygon=True)
    cx, cy, radius = compute_mec_fast(p)
    initial = np.r_[p.ravel(), cx, cy, radius]
    equality = np.zeros((5, 48))
    equality[0, 0] = equality[1, 1] = equality[2, 2] = 1
    for row, (i,j) in enumerate(((1,5),(6,10)), start=3):
        equality[row,3*i+2] = 1
        equality[row,3*j+2] = -1
    equality_value = equality @ initial
    last = None
    gaps = jacobian = None

    def evaluate(v):
        nonlocal last, gaps, jacobian
        if last is not None and np.array_equal(v, last):
            return gaps, jacobian
        centered = v[:45].reshape(15,3).copy()
        centered[:,:2] -= v[45:47]
        gaps, jac = constraint.evaluate(np.r_[centered.ravel(),v[-1]])
        placement_jac = jac[:,:45]
        jacobian = np.column_stack((placement_jac,
            -placement_jac[:,0::3].sum(axis=1),
            -placement_jac[:,1::3].sum(axis=1),jac[:,-1]))
        last = v.copy()
        return gaps,jacobian

    objective_jac = np.r_[np.zeros(47),1.]
    result = minimize(lambda v:v[-1], initial, jac=lambda v:objective_jac,
        method="SLSQP", constraints=[
            {"type":"ineq","fun":lambda v:evaluate(v)[0],"jac":lambda v:evaluate(v)[1]},
            {"type":"eq","fun":lambda v:equality@v-equality_value,"jac":lambda v:equality},
        ], options={"maxiter":maxiter,"ftol":2e-13})
    result.max_violation = float(max(0.,-evaluate(result.x)[0].min()))
    result.equality_error = float(np.abs(equality@result.x-equality_value).max())
    result.geometry_calls = constraint.calls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="experiments/lll_originalbranch_best.json")
    parser.add_argument("--output",type=Path,default=Path("experiments/best_phase_seed.json"))
    parser.add_argument("--allowance",type=float,default=1e-6)
    parser.add_argument("--maxiter",type=int,default=400)
    args=parser.parse_args()
    started=time.monotonic()
    p=load(args.input)
    result=phase_polish(p,args.allowance,args.maxiter)
    candidate=result.x[:45].reshape(15,3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps([dict(x=float(x),y=float(y),theta=float(t))
                                      for x,y,t in candidate],indent=2)+"\n")
    report=dict(radius=float(result.x[-1]),center=result.x[45:47].tolist(),
        success=bool(result.success),message=result.message,iterations=result.nit,
        max_violation=result.max_violation,equality_error=result.equality_error,
        theta0=float(candidate[0,2]),opposed_differences=[float(candidate[i,2]-candidate[j,2])for i,j in((1,5),(6,10))],
        max_polygon_area=float(polygon_areas(candidate).max()),
        movement=float(np.abs(candidate-p).max()),geometry_calls=result.geometry_calls,
        seconds=time.monotonic()-started)
    args.output.with_name(args.output.stem+"_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
