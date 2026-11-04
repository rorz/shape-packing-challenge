"""Conservative support encoding with shared XY and angle-only categorical vars."""
from __future__ import annotations
import math
import numpy as np

BIG_SCALE=10**16


def prepare(domains):
    result=[]
    for d in domains:
        grid=np.rint(d[:,:2]*1e6).astype(np.int64)
        if not np.allclose(grid*1e-6,d[:,:2],atol=5e-14,rtol=0):raise ValueError('Positions must be six-decimal grid values')
        minimum=grid.min(0);delta=grid-minimum
        angles,mapping=np.unique(d[:,2],return_inverse=True)
        result.append(dict(minimum=minimum,delta=delta,angles=angles,angle_map=mapping))
    return result


def factor(cut,prepared,scale=BIG_SCALE):
    if scale%cut['scale']:raise ValueError('Support scale must divide the factorization scale')
    directions=np.asarray(cut['directions']);weights=np.asarray(cut['weights'])
    owners=np.asarray(cut['direction_pieces']);anchor=np.asarray(cut['anchor'])
    pieces={};constant=0;worst=0;loss=0
    for i in sorted(cut['coefficients']):
        d=prepared[i];selected=(owners==i)&(weights>0)
        uu,ww=directions[selected],weights[selected]
        combined=ww@uu
        # Delta coordinates are nonnegative, so flooring signed coefficients
        # preserves a lower bound even for negative direction components.
        xy=np.floor(combined*1e-6*scale).astype(np.int64)
        position_constant=int(np.floor(float(combined@(d['minimum']*1e-6-anchor))*scale))
        theta=d['angles'];co,sn=np.cos(theta),np.sin(theta);unit=np.zeros(len(theta))
        for u,w in zip(uu,ww):
            h=np.where(co*u[0]+sn*u[1]>=0,1.,np.abs(-sn*u[0]+co*u[1]))
            unit+=w*h
        values=np.floor(unit*scale).astype(np.int64)
        item=dict(xy=xy,position_constant=position_constant,unit=values)
        pieces[i]=item;constant+=position_constant
        maximum=d['delta'].max(0)
        worst+=sum(abs(int(a))*int(b)for a,b in zip(xy,maximum))+max(abs(int(values.min())),abs(int(values.max())))
        loss+=int(maximum.sum())+2
    ratio=scale//cut['scale']
    offset=int(math.ceil((cut['center_bound']*cut['residual']+1e-11)*scale))+ratio
    worst+=abs(constant)+offset+10*scale
    if worst>=2**62:raise OverflowError('Support expression lacks conservative int64 margin')
    return dict(pieces=pieces,constant=constant,ratio=ratio,offset=offset,scale=scale,
                int64_activity_bound=worst,coefficient_floor_loss_bound=loss/scale)


def piece_values(factored,prepared,piece):
    p=prepared[piece];f=factored['pieces'][piece]
    return (f['position_constant']+p['delta']@f['xy']+f['unit'][p['angle_map']])


def audit(domains,cuts):
    prepared=prepare(domains);theta_entries=0;linear=0;max_activity=0;max_floor_loss=0
    max_above_true=-float('inf');min_cached_gap=float('inf');max_cached_gap=-float('inf')
    for cut in cuts:
        f=factor(cut,prepared);max_activity=max(max_activity,f['int64_activity_bound'])
        max_floor_loss=max(max_floor_loss,f['coefficient_floor_loss_bound'])
        nonconstant=0
        for i,item in f['pieces'].items():
            if np.any(item['unit']!=item['unit'][0]):theta_entries+=len(item['unit']);nonconstant+=1
            values=piece_values(f,prepared,i)/f['scale']
            truth=np.zeros(len(domains[i]));p=domains[i];co,sn=np.cos(p[:,2]),np.sin(p[:,2])
            for owner,u,w in zip(cut['direction_pieces'],cut['directions'],cut['weights']):
                if owner!=i or w<=0:continue
                h=np.where(co*u[0]+sn*u[1]>=0,1.,np.abs(-sn*u[0]+co*u[1]))
                truth+=w*((p[:,:2]-cut['anchor'])@u+h)
            max_above_true=max(max_above_true,float(np.max(values-truth)))
            old=cut['coefficients'][i]/cut['scale'];gap=values-old
            min_cached_gap=min(min_cached_gap,float(gap.min()));max_cached_gap=max(max_cached_gap,float(gap.max()))
        linear+=nonconstant==0
    if max_above_true>3e-14:raise AssertionError('Factorized piece exceeds exact support beyond arithmetic allowance')
    return dict(cuts=len(cuts),pure_linear_cuts=int(linear),theta_table_entries=theta_entries,
                shared_pose_table_entries=3*sum(len(d)for d in domains),
                max_int64_activity=max_activity,int64_limit=2**63-1,
                max_coefficient_floor_loss=max_floor_loss,max_above_true=max_above_true,
                cached_piece_gap_min=min_cached_gap,cached_piece_gap_max=max_cached_gap,
                scale=BIG_SCALE,roundoff_guard=1e-11,radius_integer_floor_guard=1e-12)


class Inserter:
    def __init__(self,domains):
        self.domains=domains;self.prepared=prepare(domains);self.models={}
        self.stats=dict(cuts=0,theta_tables=0,theta_table_entries=0,shared_pose_table_entries=0)
    def __call__(self,model,variables,radius,cut,serial):
        key=id(model)
        if key not in self.models:
            shared=[]
            for i,d in enumerate(self.prepared):
                maximum=d['delta'].max(0)
                x=model.new_int_var(0,int(maximum[0]),f'sep_x_{serial}_{i}')
                y=model.new_int_var(0,int(maximum[1]),f'sep_y_{serial}_{i}')
                angle=model.new_int_var(0,len(d['angles'])-1,f'sep_angle_{serial}_{i}')
                model.add_element(variables[i],d['delta'][:,0].tolist(),x)
                model.add_element(variables[i],d['delta'][:,1].tolist(),y)
                model.add_element(variables[i],d['angle_map'].tolist(),angle)
                shared.append((x,y,angle))
            # Retain model itself, preventing Python ID reuse between controls.
            self.models[key]=(model,shared)
            self.stats['shared_pose_table_entries']+=3*sum(len(d)for d in self.domains)
        shared=self.models[key][1];f=factor(cut,self.prepared)
        terms=[f['constant']]
        for i,item in f['pieces'].items():
            x,y,angle=shared[i];ax,ay=map(int,item['xy'])
            terms.extend((ax*x,ay*y));unit=item['unit']
            if np.all(unit==unit[0]):terms.append(int(unit[0]))
            else:
                value=model.new_int_var(int(unit.min()),int(unit.max()),f'sep_unit_{serial}_{i}')
                model.add_element(angle,unit.tolist(),value);terms.append(value)
                self.stats['theta_tables']+=1;self.stats['theta_table_entries']+=len(unit)
        model.add(sum(terms)<=radius*f['ratio']+f['offset'])
        self.stats['cuts']+=1
