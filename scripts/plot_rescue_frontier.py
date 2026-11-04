"""Plot exact archive overlap deficits against radius, and integer branch costs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('experiments/2026-11-04/round3')
    data=json.loads((root/'rescue-frontier/summary.json').read_text())
    rows=[r for r in data['records'] if 'exact_maximum_overlap_deficit_nm' in r]
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    ax.scatter([r['radius_above_leader_nm']for r in rows],
               [r['exact_maximum_overlap_deficit_nm']for r in rows],
               s=16,alpha=.18,color='#64748b',label='Archived states checked with GEOS')
    frontier=data['frontier']
    ax.plot([r['radius_above_leader_nm']for r in frontier],
            [r['exact_maximum_overlap_deficit_nm']for r in frontier],
            'o-',color='#087f8c',lw=1.8,label='Frontier among GEOS-checked states')
    for needle,label,offset in [('/tied/winning-section/','Three-overlap rescue',(-105,34)),
                                ('/neighbor-g/','One-overlap rescue',(-130,24)),
                                ('candidate-2.948526962','Valid incumbent',(-107,-23)),
                                ('/mirror/campaign-a/nearest','Five-overlap alternative',(-5,20))]:
        r=next(r for r in frontier if needle in r['sources'][0])
        ax.annotate(label,(r['radius_above_leader_nm'],r['exact_maximum_overlap_deficit_nm']),
                    xytext=offset,textcoords='offset points',fontsize=10,
                    arrowprops=dict(arrowstyle='-',color='#475569'))
    ax.axvline(0,color='#be123c',ls='--',lw=1,label='First-place radius')
    ax.axhline(0,color='#334155',lw=.8)
    ax.set(xlim=(-65,65),ylim=(-3,45),xlabel='Radius minus first place (billionths)',
           ylabel='Largest remaining overlap, normalized to length (billionths)',
           title='Repair targets hidden by a single combined score')
    ax.text(.02,.97,'Lower and farther left is better. Only zero overlap is valid.',
            transform=ax.transAxes,va='top',fontsize=10,color='#475569')
    ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.15)
    fig.savefig(root/'rescue-frontier.png',dpi=180);plt.close(fig)

    branches=json.loads((root/'mirror/branch-map/branches.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(s1,s2) in zip(axes.ravel(),[(1,1),(1,-1),(-1,1),(-1,-1)]):
        subset=[r for r in branches if r['first_sign']==s1 and r['second_sign']==s2]
        matrix=np.full((9,17),np.nan)
        for r in subset:matrix[r['second_offset_l']+4,r['first_offset_k']+8]=r['cost_nm']
        im=ax.imshow(matrix,origin='lower',extent=(-8.5,8.5,-4.5,4.5),
                     vmin=0,vmax=50,cmap='viridis_r',aspect='auto')
        failed=[r for r in subset if not r['feasible_positive_stress']]
        ax.scatter([r['first_offset_k']for r in failed],[r['second_offset_l']for r in failed],
                   marker='x',s=14,lw=.7,color='white')
        ax.set(title=f'Raw angle-difference signs: {s1:+d}, {s2:+d}',
               xlabel='First flat-pair branch offset (microradians)',
               ylabel='Second branch offset (microradians)',xticks=range(-8,9,2),yticks=range(-4,5,2))
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Continuous radius cost (billionths)',shrink=.8)
    fig.suptitle('612 feasible integer-angle sections; white crosses were wrongly rejected\n'
                 'by the unconstrained stress test (all pass the constrained test)',fontsize=13)
    fig.savefig(root/'branch-costs.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
