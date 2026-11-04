"""Full per-phase support bundles preserve geometric lower-bound semantics."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_phase_pool import phase_domains
from experiment_phase_bundle_resume import all_phase_supports
from semicircle_packing.fast_mec import compute_mec_fast


def test_all_balanced_phase_supports_bound_mixed_choices():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    base=np.array([[d['x'],d['y'],d['theta']]for d in data])
    domains,anchors,circles,*_=phase_domains(base,3,50e-6,1,1,100)
    cuts=all_phase_supports(domains,anchors,circles)
    assert len(cuts)>55
    rng=np.random.default_rng(110453)
    for choice in [np.zeros(15,dtype=int),*[rng.integers(0,len(domains[0]),15)for _ in range(12)]]:
        p=np.array([d[k]for d,k in zip(domains,choice)]);radius=compute_mec_fast(p)[2]
        for cut in cuts:
            lower=(sum(v[choice[i]]for i,v in cut['coefficients'].items())-cut['offset'])/cut['scale']
            assert lower<=radius+1e-11
