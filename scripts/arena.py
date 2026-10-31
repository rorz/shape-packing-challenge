#!/usr/bin/env python3
"""Read the public leaderboard or validate a solution without submitting it.

    uv run python scripts/arena.py leaderboard
    uv run python scripts/arena.py check solution.json
    uv run python scripts/arena.py check solution.json --save experiments/check.json

Uses curl because the site's edge rejects Python's default HTTP client.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess


BASE_URL = "https://www.paradigm.xyz/puzzles/api/packing"


def request(endpoint: str, payload: dict | None = None) -> dict | list:
    """Call a read/check endpoint. This module has no submission operation."""
    if endpoint not in {"leaderboard", "validate", "submissions"}:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    command = ["curl", "--silent", "--show-error", "--fail-with-body", "--max-time", "60"]
    body = None
    if payload is not None:
        command += ["--header", "Content-Type: application/json", "--data-binary", "@-"]
        body = json.dumps(payload, allow_nan=False)
    command.append(f"{BASE_URL}/{endpoint}")
    result = subprocess.run(command, input=body, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Arena request failed: {result.stderr.strip()} {result.stdout[:500]}")
    return json.loads(result.stdout)


def leaderboard() -> dict:
    return request("leaderboard")


def check_solution(solution: list[dict]) -> dict:
    """Validate with the arena's exact scoring implementation; does not publish."""
    if not isinstance(solution, list) or len(solution) != 15:
        raise ValueError("Expected a JSON array containing 15 semicircles")
    for index, placement in enumerate(solution):
        for key in ("x", "y", "theta"):
            value = placement.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Semicircle {index}: {key} must be a finite number")
    return request("validate", {"semicircles": solution})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    board_parser = subparsers.add_parser("leaderboard", help="Fetch exact scores")
    board_parser.add_argument("--save", type=Path)
    check_parser = subparsers.add_parser("check", help="Check a solution without publishing it")
    check_parser.add_argument("solution", type=Path)
    check_parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    if args.command == "leaderboard":
        result = leaderboard()
        record = {"checked_at": datetime.now(timezone.utc).isoformat(), "response": result}
    else:
        raw = args.solution.read_bytes()
        result = check_solution(json.loads(raw))
        record = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "solution": str(args.solution),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "response": result,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
