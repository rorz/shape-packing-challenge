"""Input-angle behavior independently checked against the public arena API."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.scoring import compute_mec


@pytest.mark.parametrize("raw,expected", [
    (-0.063, -0.063),
    (round(-0.063 + 2*math.pi, 6), -0.063),
    (-0.063 + 2*math.pi + .3e-6, -0.062999),
    (math.pi, -3.141592),
    (-math.pi, 3.141592),
    (3*math.pi, -3.141593),
    (-3*math.pi, 3.141593),
])
def test_angle_round_wrap_round(raw, expected):
    assert canonicalize_theta(raw) == expected


def test_cli_loader_matches_server_pi_boundary_probe(tmp_path):
    # A safe line packing makes the last flat edge determine the radius.
    # The server scores raw pi as 21.02379636805364; skipping its angle
    # normalization gives 21.023796214637695 instead.
    placements = [dict(x=3.*i, y=0., theta=0.) for i in range(15)]
    placements[-1]["theta"] = math.pi
    path = tmp_path / "packing.json"
    path.write_text(json.dumps(placements))
    spec = importlib.util.spec_from_file_location("packing_cli", Path(__file__).parents[1] / "run.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    loaded = cli._load_solution(str(path))
    assert loaded[-1].theta == -3.141592
    assert compute_mec(loaded)[2] == pytest.approx(21.02379636805364, abs=1e-12)
