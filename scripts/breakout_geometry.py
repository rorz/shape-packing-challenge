"""Fast batch scores; unsupported MEC supports fall back to the convex solver."""
from __future__ import annotations
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from semicircle_packing.fast_mec import compute_mec_fast

@lru_cache(maxsize=1)
def kernel():
    source=Path(__file__).with_name('_breakout_geometry.cpp')
    other=source.parent.parent/'src/semicircle_packing/area_kernel.cpp'
    digest=hashlib.sha256(source.read_bytes()+other.read_bytes()).hexdigest()[:16]
    path=Path(tempfile.gettempdir())/f'packing-breakout-{digest}.so'
    if not path.exists():
        temporary=path.with_suffix(f'.{os.getpid()}.tmp.so')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        temporary.replace(path)
    library=ctypes.CDLL(str(path));function=library.packing_breakout_batch
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    function.argtypes=[ctypes.c_int,ctypes.c_int,array,array,ctypes.c_double,array,array,integer]
    function.restype=None
    return library,function

def evaluate_batch(placements,centers=None,window=.01):
    p=np.ascontiguousarray(placements,dtype=float)
    if p.ndim!=3 or p.shape[2]!=3 or p.shape[1]<1 or not np.isfinite(p).all():raise ValueError('Expected finite (batch,n,3) placements with n >= 1')
    if not np.isfinite(window) or window<=0:raise ValueError('Support window must be positive and finite')
    count,n,_=p.shape
    if centers is None:centers=np.mean(p[:,:,:2],axis=1)
    centers=np.ascontiguousarray(np.broadcast_to(centers,(count,2)))
    if not np.isfinite(centers).all():raise ValueError('Centers must be finite')
    circles=np.empty((count,3));areas=np.empty((count,n*(n-1)//2));ok=np.empty(count,dtype=np.int32)
    kernel()[1](count,n,p,centers,float(window),areas,circles,ok)
    for index in np.flatnonzero(ok==0):circles[index]=compute_mec_fast(p[index],initial_center=centers[index])
    return areas,circles,ok
