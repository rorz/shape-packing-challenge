"""Audit exact decimal-grid XY recovery, modulo a common translation only."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from semicircle_packing.polish import load


def key(path):
    p=load(path)
    if p.shape!=(15,3):raise ValueError('Not a15-piece placement')
    units=np.rint(p[:,:2]*1e6).astype(np.int64)
    units-=units[0].copy()
    return hashlib.sha256(units.tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',default='experiments/2026-11-04/round3/slack-patterns/patterns.npz');ap.add_argument('--mirror',type=Path,default=Path('experiments/2026-11-04/round3/mirror'));ap.add_argument('--groups',nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data=np.load(a.training);training={};known={};raw_training=set()
    for label,path in zip(data['source_labels'],data['source_paths']):
        if label!='valid':continue
        training.setdefault(key(path),[]).append(str(path));raw_training.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    for path in a.mirror.rglob('*.json'):
        if path.relative_to(a.mirror).parts[0]in a.groups:continue
        try:k=key(path)
        except (ValueError,TypeError,KeyError,IndexError):continue
        known.setdefault(k,[]).append(str(path))
    groups={}
    for group in a.groups:
        folder=a.mirror/group;records=[]
        for path in sorted(folder.glob('elite-*.json')):
            k=key(path);records.append(dict(path=str(path),xy_hash=k,training_xy_match=training.get(k,[]),prior_xy_match=known.get(k,[]),raw_sha_training_match=hashlib.sha256(path.read_bytes()).hexdigest()in raw_training))
        unique={r['xy_hash']:r for r in records}
        manifest=folder/'frontier/manifest.json';frontier=[]
        if manifest.exists():
            info=json.loads(manifest.read_text());entries={r['file']:r for bins in info['bins'].values()for r in bins.values()}
            for entry in entries.values():
                if entry['max_overlap_nm']>=20:continue
                path=folder/'frontier'/entry['file'];k=key(path);frontier.append(entry|dict(xy_hash=k,training_xy_match=training.get(k,[]),prior_xy_match=known.get(k,[])))
        groups[group]=dict(saved_valid_records=len(records),unique_valid_xy=len(unique),training_xy_recovery=sum(bool(r['training_xy_match'])for r in unique.values()),prior_xy_recovery=sum(bool(r['prior_xy_match'])for r in unique.values()),new_xy_against_training=sum(not r['training_xy_match']for r in unique.values()),new_xy_against_prior=sum(not r['prior_xy_match']for r in unique.values()),raw_sha_training_matches=sum(r['raw_sha_training_match']for r in records),records=records,frontier_under20nm=frontier)
    result=dict(definition='Exact integer-micro XY centers, shape indices fixed, quotient common translation. Angle differences ignored. No rotation/reflection/permutation quotient.',training_unique_xy=len(training),prior_mirror_unique_xy=len(known),excluded_groups=a.groups,groups=groups)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items()if x not in ['records','frontier_under20nm']}for k,v in groups.items()},indent=2))

if __name__=='__main__':main()
