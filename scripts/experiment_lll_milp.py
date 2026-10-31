"""Mixed-integer nonlinear contact search in a reduced joint-move basis."""
from __future__ import annotations

import os
for key in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from scipy.optimize._highspy import _core as highs_core

from arena import check_solution
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, checked, save


LEADER = 2.9485269238856113
SCALE = 1e-9


def warm_milp(cost, integrality, lower, upper, matrix, row_lower, row_upper, warm, seconds):
    """Use the bundled HiGHS binding to supply an actual feasible incumbent."""
    sparse = csc_matrix(matrix)
    model = highs_core.HighsLp()
    model.num_col_ = len(cost)
    model.num_row_ = len(row_lower)
    model.col_cost_, model.col_lower_, model.col_upper_ = cost, lower, upper
    model.row_lower_, model.row_upper_ = row_lower, row_upper
    model.integrality_ = [highs_core.HighsVarType.kInteger if value else highs_core.HighsVarType.kContinuous
                          for value in integrality]
    model.a_matrix_.num_col_ = len(cost)
    model.a_matrix_.num_row_ = len(row_lower)
    model.a_matrix_.format_ = highs_core.MatrixFormat.kColwise
    model.a_matrix_.start_ = sparse.indptr.astype(np.int32)
    model.a_matrix_.index_ = sparse.indices.astype(np.int32)
    model.a_matrix_.value_ = sparse.data
    solver = highs_core._Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("time_limit", seconds)
    solver.setOptionValue("mip_rel_gap", 0.)
    solver.setOptionValue("threads", 1)
    solver.passModel(model)
    solver.setSolution(len(warm), np.arange(len(warm), dtype=np.int32), warm)
    solver.run()
    solution = solver.getSolution()
    status = solver.getModelStatus()
    return SimpleNamespace(x=np.array(solution.col_value) if solution.value_valid else None,
                           status=2 if status == highs_core.HighsModelStatus.kInfeasible else 0,
                           message=solver.modelStatusToString(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/lll_hillclimb_best_state.npz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/lll_milp.json"))
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--solve-seconds", type=float, default=2)
    parser.add_argument("--coefficients", type=int, default=3)
    parser.add_argument("--movement", type=float, default=500, help="Maximum absolute placement step in micro-units")
    parser.add_argument("--margin", type=float, default=0, help="Required clearance in nanometers")
    parser.add_argument("--relax", action="store_true", help="Minimize maximum violation before reaching exact feasibility")
    parser.add_argument("--target", type=float)
    parser.add_argument("--radius-only-slack", action="store_true", help="Keep overlap constraints hard while minimizing enclosing radius")
    args = parser.parse_args()
    with np.load(args.checkpoint) as state:
        base = np.round(state["placements"], 6)
        origin = state["origin"].copy()
        moves = np.rint(state["moves"]).astype(np.int64).reshape(45, 45)
        target = float(state["target"]) if args.target is None else args.target
    constraint = AreaConstraints(origin)
    # Coefficients multiply rows of the reduced basis. Each such integer
    # combination remains exactly on the original six-decimal lattice.
    direction = moves.T.astype(float) * 1e-6
    matrix_rows, lower_rows, upper_rows = [], [], []
    radius_rows = []
    started = time.monotonic()
    best_violation = float("inf")
    best_loss = float("inf")
    iteration = 0
    solve_budget = args.solve_seconds
    seen = set()

    def evaluate(variables):
        coefficients = variables[:45]
        candidate = np.round(base + (coefficients @ moves).reshape(15, 3) * 1e-6, 6)
        candidate[:, 2] = [canonicalize_theta(theta) for theta in candidate[:, 2]]
        centered = candidate.copy()
        centered[:, :2] -= variables[45:47] * SCALE
        gaps, jac = constraint.evaluate(np.r_[centered.ravel(), target])
        placement_jac = jac[:, :-1]
        small_jac = np.column_stack((placement_jac @ direction,
            -placement_jac[:, 0::3].sum(axis=1) * SCALE,
            -placement_jac[:, 1::3].sum(axis=1) * SCALE))
        return candidate, gaps, small_jac

    def add_cuts(coefficients, gaps, jac):
        # Keep every possibly active contact, including both flat endpoints.
        active = (gaps < max(.001, args.movement * 10e-6)) & (np.linalg.norm(jac, axis=1) > 1e-15)
        rows = jac[active] / SCALE
        matrix_rows.append(rows)
        radius_rows.append((np.flatnonzero(active) >= 105).astype(float))
        lower_rows.append((args.margin * SCALE - gaps[active] + jac[active] @ coefficients) / SCALE)
        upper_rows.append(np.full(active.sum(), np.inf))

    cx, cy, _ = compute_mec_fast(base)
    coefficients = np.r_[np.zeros(45), cx/SCALE, cy/SCALE]
    initial_variables = coefficients.copy()
    candidate, gaps, jac = evaluate(coefficients)
    add_cuts(coefficients, gaps, jac)
    # Limit nonlinear displacement while allowing combinations of large soft
    # basis directions to cancel each other.
    matrix_rows.append(np.column_stack((moves.T.astype(float), np.zeros((45, 2)))))
    radius_rows.append(np.zeros(45))
    lower_rows.append(np.full(45, -args.movement))
    upper_rows.append(np.full(45, args.movement))
    while time.monotonic() - started < args.seconds:
        all_rows = np.vstack(matrix_rows)
        objective = np.zeros(47)
        integrality = np.r_[np.ones(45), np.zeros(2)]
        bound_lower = np.r_[np.full(45, -args.coefficients, dtype=float), np.full(2, -args.movement*1000)]
        bound_upper = np.r_[np.full(45, args.coefficients, dtype=float), np.full(2, args.movement*1000)]
        for index, move in enumerate(moves.reshape(45,15,3)):
            if np.all(move[:,2] == 0) and np.all(move[:,:2] == move[0,:2]):
                bound_lower[index] = bound_upper[index] = 0
        if args.relax:
            if args.radius_only_slack:
                slack = np.concatenate(radius_rows)
            else:
                slack = np.ones(len(all_rows))
                # The physical displacement rows have no relaxation.
                offset = len(matrix_rows[0])
                slack[offset:offset+45] = 0
            all_rows = np.column_stack((all_rows, slack))
            objective = np.r_[objective, 1.0]
            integrality = np.r_[integrality, 0]
            bound_lower = np.r_[bound_lower, 0.0]
            bound_upper = np.r_[bound_upper, 1e6]
        row_lower, row_upper = np.concatenate(lower_rows), np.concatenate(upper_rows)
        seconds = min(solve_budget, max(.01,args.seconds-(time.monotonic()-started)))
        if args.relax:
            warm = np.r_[initial_variables, 0.0]
            residual = row_lower - all_rows @ warm
            warm[-1] = max(0., float(residual[slack > 0].max())) + 1e-7
            result = warm_milp(objective, integrality, bound_lower, bound_upper,
                               all_rows, row_lower, row_upper, warm, seconds)
        else:
            result = milp(objective, integrality=integrality,
                bounds=Bounds(bound_lower, bound_upper),
                constraints=LinearConstraint(all_rows, row_lower, row_upper),
                options={"time_limit": seconds, "mip_rel_gap": 0.0})
        iteration += 1
        if result.x is None:
            print(json.dumps(dict(iteration=iteration, event="no_candidate", message=result.message,
                                  seconds=time.monotonic()-started)), flush=True)
            if result.status == 2:
                return
            solve_budget *= 2
            continue
        coefficients = result.x[:47].copy()
        coefficients[:45] = np.rint(coefficients[:45])
        candidate, gaps, jac = evaluate(coefficients)
        violation = float(-gaps.min())
        negative = np.minimum(gaps, 0)
        loss = float(negative @ negative)
        print(json.dumps(dict(iteration=iteration, violation=violation, loss=loss,
                              max_coeff=float(np.abs(coefficients[:45]).max()),
                              linear_slack_nm=float(result.x[-1]) if args.relax else 0,
                              movement=float(np.abs(coefficients[:45] @ moves).max()),
                              seconds=time.monotonic()-started)), flush=True)
        if violation < best_violation or loss < best_loss:
            best_violation = min(best_violation, violation)
            best_loss = min(best_loss, loss)
            save(args.output.with_name(args.output.stem + "_nearest.json"), candidate)
            snapshot = args.output.with_name(args.output.stem + "_state.npz")
            np.savez(snapshot, placements=candidate, origin=origin,
                     moves=moves.reshape(45,15,3), target=target,
                     loss=loss, violation=violation)
        if violation < 2e-9 or float(gaps[:105].min()) > -1e-8:
            validation = checked(candidate)
            print(json.dumps(dict(event="reference", valid=validation.valid,
                                  score=validation.score, errors=validation.errors)), flush=True)
            if validation.valid and validation.score < LEADER:
                save(args.output, candidate)
                response = check_solution([dict(x=float(x), y=float(y), theta=float(t)) for x,y,t in candidate])
                args.output.with_name(args.output.stem + "_api.json").write_text(json.dumps(response, indent=2) + "\n")
                print(json.dumps(dict(event="api", **response)), flush=True)
                if response["valid"] and response["score"] < LEADER:
                    return
        key = tuple(coefficients[:45])
        if key in seen:
            solve_budget *= 2
        else:
            add_cuts(coefficients, gaps, jac)
            seen.add(key)
            solve_budget = args.solve_seconds


if __name__ == "__main__":
    main()
