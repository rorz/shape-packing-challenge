"""Run a blind worker from a source-only bundle with audited problem reads."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-repo", type=Path, required=True)
    parser.add_argument("--method", choices=("native", "hybrid"), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exact-seconds", type=float, nargs=3, default=(2, 4, 10))
    args = parser.parse_args()
    repo = str(args.original_repo.resolve()) + os.sep
    environment = repo + ".venv" + os.sep
    output = str(args.output.resolve()) + os.sep
    audit = {"json_reads": set(), "blocked_reads": [], "network_attempts": 0}

    def hook(event, values):
        if event == "socket.connect":
            audit["network_attempts"] += 1
            raise PermissionError("Blind workers are offline")
        if event != "open" or not values or isinstance(values[0], int):
            return
        path = os.path.abspath(os.fsdecode(values[0]))
        mode, flags = values[1], values[2]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
        if not reading:
            return
        if path.startswith(repo) and not path.startswith((environment, output)):
            audit["blocked_reads"].append(path)
            raise PermissionError(f"Original workspace data is unavailable to the blind worker: {path}")
        if path.endswith(".json"):
            audit["json_reads"].add(path)

    sys.addaudithook(hook)
    try:
        import blind_packing_worker
        # A separate guard catches accidental API calls even before networking.
        def no_api(*unused_args, **unused_kwargs):
            raise PermissionError("Blind workers must not call the arena API")
        import arena
        import lazy_phase_solver
        arena.check_solution = no_api
        lazy_phase_solver.check_solution = no_api
        blind_packing_worker.run(args.problem, args.output, method=args.method,
                                seconds=args.seconds, seed=args.seed,
                                exact_seconds=tuple(args.exact_seconds))
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {**audit, "json_reads": sorted(audit["json_reads"]),
                  "scope": "Python file/network audit; source bundle omits all held-out targets and fitted frames"}
        (args.output / "access-audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
