"""Reproducible geometry and slack diagnostic for the saved packing."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/packing-mpl')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse
import itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from semicircle_packing.polish import AreaConstraints,load,checked
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='solution.json')
    ap.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/round2/contact-slack.png'))
    ap.add_argument('--leader',type=float,default=2.9485269238856113);a=ap.parse_args()
    raw=load(a.input);reference=checked(raw)
    p=np.round(raw,6);p[:,2]=[canonicalize_theta(t)for t in p[:,2]]
    cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy]
    constraints=AreaConstraints(p);g,j=constraints.evaluate(np.r_[p.ravel(),r]);gap=(r-a.leader)*1e9
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,8),facecolor='#fafaf8');grid=fig.add_gridspec(2,2,width_ratios=[1.2,1],hspace=.34,wspace=.25)
    ax=fig.add_subplot(grid[:,0]);ax.set_facecolor('#fafaf8')
    colors=plt.get_cmap('Set3')
    for i,(x,y,t) in enumerate(p):
        arc=np.linspace(t-np.pi/2,t+np.pi/2,200);vertices=np.c_[x+np.cos(arc),y+np.sin(arc)]
        ax.add_patch(Polygon(vertices,closed=True,facecolor=colors(i%12),edgecolor='#314047',linewidth=1))
        ax.text(x+.43*np.cos(t),y+.43*np.sin(t),str(i),ha='center',va='center',weight='bold',color='#17252c')
    ax.add_patch(Circle((0,0),r,fill=False,color='#17252c',lw=1.8))
    pairs=list(itertools.combinations(range(15),2));tight=np.flatnonzero(g[:105]<50e-9)
    for k in tight:
        i,h=pairs[k];ax.plot(p[[i,h],0],p[[i,h],1],color='#b42c36',lw=1.6,ls='--',alpha=.8)
    support_rows=np.flatnonzero(g[105:]<1e-10)
    for row in support_rows:
        i,feature=divmod(int(row),3);x,y,t=p[i]
        if feature==2:far=p[i,:2]*(1+1/np.linalg.norm(p[i,:2]))
        else:
            sign=-1 if feature==0 else 1
            far=np.array([x-sign*np.sin(t),y+sign*np.cos(t)])
        ax.scatter(*far,s=52,c='#17252c',zorder=8)
    endpoint_sensitive=[i for i in range(len(p))if max(g[105+3*i:107+3*i])<1e-6]
    for i in endpoint_sensitive:ax.scatter(*p[i,:2],s=48,facecolors='none',edgecolors='#b42c36',zorder=8)
    ax.set(xlim=(-3.17,3.17),ylim=(-3.17,3.17),aspect='equal');ax.axis('off')
    ax.set_title('Current packing and its tightest pair constraints',loc='left',fontsize=14,pad=10)
    support_labels=' / '.join(map(str,sorted({int(row)//3 for row in support_rows})))
    ax.text(0,-3.40,f'Black dots: circle supports {support_labels}\nDashed red: pair slack below 50 billionths; labels are zero-based',ha='center',va='top',fontsize=10,color='#45545a')
    axb=fig.add_subplot(grid[0,1]);axb.set_facecolor('#fafaf8')
    ids=np.flatnonzero(g[105:]<1e-6);slack=g[105+ids]*1e9;labels=[f'{i//3} '+('endpoint −','endpoint +','arc')[i%3]for i in ids]
    axb.barh(labels,slack,color=['#b42c36'if s<gap else '#7ca6b3'for s in slack]);axb.invert_yaxis()
    axb.axvline(gap,color='#17252c',ls='--',lw=1,label=f'Lead gap: {gap:.2f}')
    axb.set(xlabel='Exact boundary slack (×10⁻⁹)',title='Several boundaries are within the lead gap')
    axb.legend(frameon=False,loc='lower right',fontsize=9)
    axc=fig.add_subplot(grid[1,1]);axc.set_facecolor('#fafaf8')
    ids=np.argsort(g[:105])[:12];labels=[f'{pairs[k][0]}–{pairs[k][1]}'for k in ids]
    axc.barh(labels,g[ids]*1e9,color='#b9847e');axc.invert_yaxis();axc.axvline(gap,color='#17252c',ls='--',lw=1)
    axc.set(xlabel='Local normalized pair slack (×10⁻⁹)',title='Moving one piece spends its neighbors’ slack')
    displayed_radius=reference.score if reference.valid else r
    fig.suptitle(f'Our radius {displayed_radius:.15f}   |   Leader {a.leader:.15f}',x=.05,ha='left',fontsize=16,weight='bold',y=.97)
    fig.text(.05,.035,'One coordinate grid step = 1,000 on these scales. Pair slack uses area / Jacobian norm; it is a local diagnostic, not an exact clearance.',fontsize=10,color='#45545a')
    fig.subplots_adjust(top=.88,bottom=.12,left=.05,right=.97);a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170,facecolor=fig.get_facecolor());fig.savefig(a.output.with_suffix('.svg'),facecolor=fig.get_facecolor())
    print(a.output)

if __name__=='__main__':main()
