"""Show the discovered three-piece reflection in a common coordinate frame."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/packing-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from semicircle_packing.polish import load
from semicircle_packing.fast_mec import compute_mec_fast
from experiment_surgery_patch import transformed

def main():
    root=Path('experiments/2026-11-04/round2/topology')
    reference=root/'reference-continuous.json'
    if not reference.exists():
        p=np.load('experiments/soft_modes.npz')['p']
        reference.write_text(json.dumps([dict(x=float(x),y=float(y),theta=float(t))for x,y,t in p],indent=2)+'\n')
    p=load(reference);q=load(root/'continuous-best.json');audit=json.loads((root/'patch-audit.json').read_text())
    outer=audit['outer'];q=transformed(q,outer['phi'],np.array(outer['translation']),outer['reflected'])
    # Reorder only for identical labels and colors across the two drawings.
    mapping=list(outer['mapping'])
    for index,source in enumerate(audit['source_patch']):
        mapping[source]=audit['target_patch'][audit['patch']['mapping'][index]]
    reordered=np.empty_like(q)
    for source,target in enumerate(mapping):reordered[target]=q[source]
    q=reordered;patch=set(audit['target_patch']);axis=np.array(audit['patch_reflection']['axis_direction'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,6.4),facecolor='#fafaf8')
    for ax,shapes,title in zip(axes,[p,q],['Original continuous arrangement','Three-piece patch reflected']):
        ax.set_facecolor('#fafaf8');cx,cy,r=compute_mec_fast(shapes)
        for i,(x,y,t) in enumerate(shapes):
            a=np.linspace(t-np.pi/2,t+np.pi/2,180);xy=np.c_[x+np.cos(a),y+np.sin(a)]
            color={1:'#e8a370',5:'#d8796b',14:'#ebc778'}.get(i,'#dce4e6')
            ax.add_patch(Polygon(xy,facecolor=color,edgecolor='#506169',lw=.8))
            if i in patch:ax.text(x+.4*np.cos(t),y+.4*np.sin(t),str(i),ha='center',va='center',weight='bold')
        ax.add_patch(Circle((cx,cy),r,fill=False,color='#263d46',lw=1.5))
        ax.plot([-3.1*axis[0],3.1*axis[0]],[-3.1*axis[1],3.1*axis[1]],'--',color='#9b3831',lw=1.2)
        ax.set(xlim=(-3.2,3.2),ylim=(-3.2,3.2),aspect='equal');ax.axis('off');ax.set_title(title,fontsize=14,pad=12)
    fig.suptitle('Same continuous radius, different rounding problem',fontsize=18,weight='bold',x=.04,ha='left',y=.97)
    fig.text(.04,.035,'Colored pieces form the reflected patch. The other twelve align within 1.1×10⁻⁶. Both are continuous seeds requiring grid repair.',fontsize=10,color='#45545a')
    fig.subplots_adjust(top=.85,bottom=.09,left=.03,right=.97,wspace=.08)
    out=root.parent/'patch-reflection.png';fig.savefig(out,dpi=170,facecolor=fig.get_facecolor());fig.savefig(out.with_suffix('.svg'),facecolor=fig.get_facecolor());print(out)

if __name__=='__main__':main()
