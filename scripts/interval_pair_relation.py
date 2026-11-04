"""Exact interval representation of a finite binary compatibility relation."""
from __future__ import annotations
import numpy as np


def quotient(mask):
 rows,row_map=np.unique(np.packbits(mask,axis=1),axis=0,return_inverse=True)
 rows=np.unpackbits(rows,count=mask.shape[1],axis=1).astype(bool)
 _,indices,column_map=np.unique(np.packbits(rows.T,axis=1),axis=0,return_index=True,return_inverse=True)
 return row_map,column_map,rows[:,indices]


def intervals(mask,order):
 """Rows (row_class, inclusive_lo, inclusive_hi) in the chosen column order."""
 ordered=mask[:,order];padded=np.pad(ordered,((0,0),(1,1)),constant_values=False)
 starts=np.column_stack(np.nonzero(padded[:,1:]&~padded[:,:-1]))
 ends=np.column_stack(np.nonzero(padded[:,:-1]&~padded[:,1:]))
 assert np.array_equal(starts[:,0],ends[:,0])
 return np.column_stack((starts[:,0],starts[:,1],ends[:,1]-1)).astype(np.int64)


def reconstruct(shape,order,segments):
 ordered=np.zeros(shape,dtype=bool)
 for row,lo,hi in segments:ordered[row,lo:hi+1]=True
 answer=np.zeros(shape,dtype=bool);answer[:,order]=ordered;return answer


def column_orders(mask,a,b):
 """Geometry/similarity heuristics change order only, never allowed tuples."""
 m=mask.shape[1];orders={'identity':np.arange(m),'degree':np.argsort(mask.sum(0),kind='stable')}
 normal=np.array([np.cos(a[:,2]).mean(),np.sin(a[:,2]).mean()]);normal/=np.linalg.norm(normal);tangent=np.array([-normal[1],normal[0]])
 n=b[:,:2]@normal;t=b[:,:2]@tangent;theta=np.unwrap(b[:,2]);theta-=theta[0];origin=.5*(np.mean(a[:,:2],axis=0)+np.mean(b[:,:2],axis=0));tc=float(origin@tangent)
 nb=np.column_stack((np.cos(b[:,2]),np.sin(b[:,2])));slope=(nb@tangent)/(nb@normal)
 values={'normal':n,'tangent':t,'theta':theta}
 for offset in (-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.):values[f'flat_{offset:+.2f}']=n-slope*(tc+offset-t)
 for name,value in values.items():
  orders[name]=np.argsort(value,kind='stable')
  if name!='theta':orders['theta_then_'+name]=np.lexsort((value,np.rint(theta*1e6)))
 # First singular direction of centered row masks via bounded power iteration.
 centered=mask.astype(float);centered-=centered.mean(0);v=np.linspace(-1,1,m)
 for _ in range(16):
  v=centered.T@(centered@v);norm=np.linalg.norm(v)
  if norm==0:break
  v/=norm
 orders['spectral']=np.argsort(v,kind='stable')
 return orders


def add_interval_constraint(model,first,second,row_map,column_map,order,segments,name='interval'):
 """Project an interval table using existential lo/hi and rank inequalities."""
 nrows=int(np.max(row_map))+1;ncols=len(order);inverse=np.empty(ncols,dtype=np.int64);inverse[order]=np.arange(ncols)
 row=model.new_int_var(0,nrows-1,name+'_row');rank=model.new_int_var(0,ncols-1,name+'_rank')
 model.add_element(first,np.asarray(row_map,dtype=int).tolist(),row);model.add_element(second,inverse[column_map].tolist(),rank)
 low=model.new_int_var(0,ncols-1,name+'_lo');high=model.new_int_var(0,ncols-1,name+'_hi')
 model.add_allowed_assignments([row,low,high],segments.tolist());model.add(rank>=low);model.add(rank<=high)
 return row,rank,low,high


def verify_element_encoding(mask,row_map,column_map,order,segments):
 """Exhaustively check actual categorical IDs through the element semantics."""
 inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order));actual_ranks=inverse[column_map]
 bounds=[segments[segments[:,0]==row,1:]for row in range(int(np.max(row_map))+1)]
 for actual_row,row_class in enumerate(row_map):
  choices=bounds[row_class]
  admitted=((actual_ranks[None,:]>=choices[:,0,None])&(actual_ranks[None,:]<=choices[:,1,None])).any(0)
  if not np.array_equal(admitted,mask[actual_row]):raise AssertionError(f'Element mapping differs at actual row{actual_row}')
 return int(mask.size)


def add_allowed_interval_pair(model,first,second,allowed,ni,nj):
 """Drop-in prototype for the master's binary relation builder, identity order."""
 mask=np.zeros((ni,nj),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True
 row_map,column_map,small=quotient(mask);order=np.arange(small.shape[1]);segments=intervals(small,order)
 add_interval_constraint(model,first,second,row_map,column_map,order,segments,f'interval_{first.index}_{second.index}')
 return [ni,nj,*small.shape]
