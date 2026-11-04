"""Transfer contact-slack allocations across fresh mirror-family grid phases.

This changes proposal distributions only. Geometry and GEOS validation remain
unchanged, and a fixed fraction of proposals retain the original Dirichlet law.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from semicircle_packing.geometry import canonicalize_theta


class SlackMixture:
    def __init__(self, path, *, fraction=.8, floor=.02, strength=0., include_near=False, quality_temperature_nm=0.):
        source=np.load(path)
        patterns=np.asarray(source['patterns'],float)
        self.active_ids=np.asarray(source['active_ids'],int)
        if patterns.ndim!=2 or patterns.shape[1]!=len(self.active_ids):
            raise ValueError('Training patterns and active row IDs disagree')
        keep=np.ones(len(patterns),bool)
        if 'valid' in source and not include_near:keep&=source['valid'].astype(bool)
        if 'source_labels' in source and not include_near:keep&=source['source_labels']=='valid'
        self.source_indices=np.flatnonzero(keep)
        patterns=patterns[keep]
        if not len(patterns) or not np.isfinite(patterns).all() or np.min(patterns)<0:
            raise ValueError('Expected finite nonnegative training patterns')
        totals=patterns.sum(axis=1)
        if np.min(totals)<=0:raise ValueError('Zero training pattern')
        self.patterns=patterns/totals[:,None]
        self.temperature=float(quality_temperature_nm)
        self.probabilities=np.full(len(patterns),1./len(patterns))
        if self.temperature:
            if self.temperature<0 or 'source_scores' not in source:raise ValueError('Quality weighting needs scores and positive temperature')
            scores=source['source_scores'][keep]
            self.probabilities=np.exp(-(scores-scores.min())*1e9/self.temperature)
            self.probabilities/=self.probabilities.sum()
        self.fraction=float(fraction);self.floor=float(floor);self.strength=float(strength)
        if not 0<=self.fraction<=1 or not 0<self.floor<=1:raise ValueError('Invalid mixture/floor')
        self.source_path=str(path);self.source_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.counts=Counter();self.source_counts=Counter()

    def allocation(self, frame, rng, count, args):
        active=np.flatnonzero(frame.active)
        if not np.array_equal(active,self.active_ids):
            raise ValueError('Fresh frame active contacts differ from training pattern order')
        alpha=np.full(len(frame.weights),args.concentration,dtype=float)
        alpha[active>=105]*=args.boundary_bias
        allocation=rng.dirichlet(alpha,size=count)
        selected=np.flatnonzero(rng.random(count)<self.fraction)
        choices=rng.choice(len(self.patterns),size=len(selected),p=self.probabilities)
        strength=self.strength or len(frame.weights)*args.concentration
        shapes=strength*((1-self.floor)*self.patterns[choices]+self.floor/len(frame.weights))
        samples=rng.gamma(shapes,1.)
        samples/=samples.sum(axis=1,keepdims=True)
        allocation[selected]=samples
        self.counts['learned']+=len(selected);self.counts['unbiased']+=count-len(selected)
        for index,frequency in zip(*np.unique(choices,return_counts=True)):
            self.source_counts[int(self.source_indices[index])]+=int(frequency)
        return allocation

    def sample(self,frame,rng,count,args,decoder):
        allocation=self.allocation(frame,rng,count,args)
        desired=frame.budget*allocation/frame.weights[None,:]/1e-6
        contact=(frame.weights[None,:]*(desired+frame.b0))@frame.projection.T
        dimension=len(frame.moves)
        wanted=rng.normal(0,args.movement,(count,dimension))
        blocks=[wanted*frame.identity,contact*frame.scale]
        if dimension==45:
            blocks.append(np.broadcast_to(frame.flat_offset,(count,len(frame.flat_offset)))*frame.scale)
        ambient=np.rint(np.column_stack(blocks)).astype(np.int64)
        transform=frame.gso_transform if dimension==43 else frame.transform
        actual,units,coeff,ambiguous=decoder.decode(np.ascontiguousarray(ambient@transform),frame.mu,frame.moves,frame.anchor_units)
        if dimension==45:
            patches=frame.patch(ambient,(actual,units,coeff,ambiguous))
        else:
            for index in np.flatnonzero(ambiguous):
                coeff[index]=frame.M.babai(ambient[index].tolist())
                raw=frame.raw(coeff[index]);u=np.rint(raw*1e6).astype(np.int64)
                u[:,2]=np.rint([canonicalize_theta(theta)*1e6 for theta in raw[:,2]]).astype(np.int64)
                units[index]=u.ravel();actual[index]=u*1e-6
            patches=int(ambiguous.sum())
        return actual,units,coeff,ambient,patches

    def report(self):
        return dict(source=self.source_path,source_sha256=self.source_hash,patterns=len(self.patterns),fraction=self.fraction,floor=self.floor,strength=self.strength or 'active_count*frame_concentration',quality_temperature_nm=self.temperature,effective_pattern_count=float(1./(self.probabilities@self.probabilities)),source_probabilities={int(k):float(p)for k,p in zip(self.source_indices,self.probabilities)},counts=dict(self.counts),source_counts=dict(self.source_counts))
