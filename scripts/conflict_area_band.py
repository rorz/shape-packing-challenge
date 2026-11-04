"""Native true semicircle area and rigorous local inscribed-polygon loss band."""
from __future__ import annotations
import ctypes,hashlib,os,subprocess,tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def kernel():
    from semicircle_packing.config import POLYGON_ARC_POINTS
    if POLYGON_ARC_POINTS!=4096:
        raise ValueError('Area loss band is derived for exactly4096 arc vertices')
    source=Path(__file__).with_name('_conflict_area_band.cpp')
    other=source.parent.parent/'src/semicircle_packing/area_kernel.cpp'
    digest=hashlib.sha256(source.read_bytes()+other.read_bytes()).hexdigest()[:20]
    output=Path(tempfile.gettempdir())/f'packing-area-band-{digest}.so'
    if not output.exists():
        temporary=output.with_suffix(f'.{os.getpid()}.tmp.so')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        temporary.replace(output)
    library=ctypes.CDLL(str(output));function=library.packing_true_area_loss_band
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    function.argtypes=[ctypes.c_int,ctypes.c_int,array,array,array,array]
    function.restype=None
    return library,function


def true_area_loss_band(a,b):
    a,b=np.ascontiguousarray(a,dtype=float),np.ascontiguousarray(b,dtype=float)
    area=np.empty((len(a),len(b)));loss=np.empty_like(area)
    kernel()[1](len(a),len(b),a,b,area,loss)
    return area,loss
