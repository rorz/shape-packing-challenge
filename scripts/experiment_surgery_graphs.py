"""Independently test unlabeled topology coverage and symmetry invariance."""
import argparse,itertools,json,hashlib
from pathlib import Path
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match
from experiment_surgery import Geometry,fingerprint
from semicircle_packing.polish import load

def graph(p,R,geometry):
 n=len(p);gaps=geometry.fun(np.r_[p.ravel(),R]);pairs=list(itertools.combinations(range(n),2));G=nx.Graph();boundary=gaps[len(pairs):].reshape(n,3)
 for i in range(n):G.add_node(i,boundary=int((boundary[i]<3e-6).sum()))
 for row,(i,j)in enumerate(pairs):
  if gaps[row]<3e-6:G.add_edge(i,j)
 return G

def cluster(graphs,colored):
 buckets={};representatives=[];classes=[];match=categorical_node_match('boundary',0)if colored else None
 for index,G in enumerate(graphs):
  key=nx.weisfeiler_lehman_graph_hash(G,node_attr='boundary'if colored else None,iterations=5,digest_size=24);bucket=buckets.setdefault(key,[]);assigned=None
  for representative,class_index in bucket:
   if nx.is_isomorphic(G,representative,node_match=match):assigned=class_index;break
  if assigned is None:assigned=len(representatives);representatives.append(G);bucket.append((G,assigned))
  classes.append(assigned)
 return len(representatives),classes,len(buckets)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='experiments/2026-11-04/round2/topology');ap.add_argument('--seed',type=int,default=811043);a=ap.parse_args();root=Path(a.root);rows=[json.loads(s)for s in(root/'trials.jsonl').read_text().splitlines()];by_signature={}
 for row in rows:
  signature=row['fingerprint'];by_signature[signature]=min(by_signature.get(signature,float('inf')),row['radius'])
 geometry=Geometry(np.ones(15));rng=np.random.default_rng(a.seed);graphs=[];paths=[];signature_mismatches=[];permutation_mismatches=[];reflection_mismatches=[]
 for path in sorted(root.glob('topology-*.json')):
  expected=path.stem.split('topology-',1)[1];p=load(path);R=by_signature[expected];actual=fingerprint(geometry,p,R)
  if actual!=expected:signature_mismatches.append(str(path))
  permutation=rng.permutation(15)
  if fingerprint(geometry,p[permutation],R)!=actual:permutation_mismatches.append(str(path))
  reflected=p.copy();reflected[:,1]*=-1;reflected[:,2]*=-1
  if fingerprint(geometry,reflected,R)!=actual:reflection_mismatches.append(str(path))
  graphs.append(graph(p,R,geometry));paths.append(str(path))
 colored,colorclasses,colored_hashes=cluster(graphs,True);plain,plainclasses,plain_hashes=cluster(graphs,False)
 report=dict(seed=a.seed,networkx_version=nx.__version__,trials=len(rows),signature_count_all_trials=len({r['fingerprint']for r in rows}),feasible_trials=sum(r['feasible']for r in rows),signature_count_feasible_trials=len({r['fingerprint']for r in rows if r['feasible']}),archived_graphs=len(graphs),colored_isomorphism_classes=colored,uncolored_isomorphism_classes=plain,colored_wl_buckets=colored_hashes,uncolored_wl_buckets=plain_hashes,original_hash_reconstruction_mismatches=signature_mismatches,permutation_invariance_mismatches=permutation_mismatches,reflection_invariance_mismatches=reflection_mismatches,contact_threshold=3e-6,boundary_label='number of active endpoint/arc boundary constraints',classes=[dict(path=p,colored=c,uncolored=u)for p,c,u in zip(paths,colorclasses,plainclasses)])
 (root/'graph-isomorphism-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='classes'},indent=2))
if __name__=='__main__':main()
