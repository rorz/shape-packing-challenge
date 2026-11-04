import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_reflection_components import components,reflected
from semicircle_packing.geometry import Semicircle,semicircle_polygon


def test_components_exactly_encode_all_subset_choices():
    rng=np.random.default_rng(110476)
    for n in range(2,9):
        for _ in range(20):
            cross=rng.integers(0,2,size=(n,n));cross=np.maximum(cross,cross.T);np.fill_diagonal(cross,0)
            groups=components(cross,.5)
            for mask in range(1<<n):
                bits=np.array([(mask>>i)&1 for i in range(n)])
                direct=not np.any((bits[:,None]!=bits[None,:])&(cross>.5))
                grouped=all(len(set(bits[g]))==1 for g in groups)
                assert direct==grouped


def test_reflection_cross_overlap_is_symmetric_and_preserves_internal_pairs():
    p=np.array([[.2,.3,.7],[.8,-.1,2.1],[-.3,.6,-1.1]])
    q=reflected(p,.85);P=[semicircle_polygon(Semicircle(*v))for v in p];Q=[semicircle_polygon(Semicircle(*v))for v in q]
    for i in range(3):
        for j in range(i):
            assert abs(P[i].intersection(Q[j]).area-Q[i].intersection(P[j]).area)<1e-13
            assert abs(P[i].intersection(P[j]).area-Q[i].intersection(Q[j]).area)<1e-13
