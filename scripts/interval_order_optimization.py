"""Native finite-order heuristic; compatibility values are never approximated."""
from __future__ import annotations
import ctypes,hashlib,os,subprocess,tempfile
from pathlib import Path
import numpy as np


def optimize(mask,initial=None,*,seconds=1.,iterations=400):
 source=Path(__file__).with_name('_interval_order.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16];path=Path(tempfile.gettempdir())/f'packing-interval-order-{digest}.so'
 if not path.exists():
  temporary=path.with_suffix(f'.{os.getpid()}.tmp.so');subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,capture_output=True);temporary.replace(path)
 library=ctypes.CDLL(str(path));u64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');library.interval_hamming_two_opt.argtypes=[ctypes.c_int,ctypes.c_int,u64,i64,ctypes.c_int,ctypes.c_double,i64,i64];library.interval_hamming_two_opt.restype=None
 n=mask.shape[1];packed=np.packbits(mask.T,axis=1);padded=np.pad(packed,((0,0),(0,(-packed.shape[1])%8)));bits=np.ascontiguousarray(padded).view(np.uint64);initial=np.ascontiguousarray(np.arange(n)if initial is None else initial,dtype=np.int64);assert np.array_equal(np.sort(initial),np.arange(n));output=np.empty(n,dtype=np.int64);stats=np.empty(3,dtype=np.int64)
 library.interval_hamming_two_opt(n,bits.shape[1],bits,initial,iterations,seconds,output,stats)
 assert np.array_equal(np.sort(output),np.arange(n));return output,dict(initial_intervals=int(stats[0]//2),final_intervals=int(stats[1]//2),two_opt_moves=int(stats[2]))


def add_hamming_interval_pair(model,first,second,allowed,ni,nj):
 """Optional separate builder; leaves the stable identity-order helper alone."""
 from interval_pair_relation import quotient,intervals,add_interval_constraint
 mask=np.zeros((ni,nj),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True;rows,cols,small=quotient(mask);order=np.arange(small.shape[1]);segments=intervals(small,order)
 if len(segments)>2*len(small) and small.shape[1]>2:
  order,_=optimize(small,order,seconds=2.,iterations=10000);segments=intervals(small,order)
 add_interval_constraint(model,first,second,rows,cols,order,segments,f'hamming_{first.index}_{second.index}')
 return [ni,nj,*small.shape]
