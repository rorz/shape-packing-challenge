"""Sweep LLL contact allocation and movement settings across nearby branches."""
import argparse,itertools,json,subprocess,time
from pathlib import Path
from semicircle_packing.polish import load,checked
from arena import check_solution

parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True);args=parser.parse_args()
root=Path('experiments/sampling_sweep');root.mkdir(parents=True,exist_ok=True)
marker=root/'winner.json';start=time.perf_counter();best=2.9485269708615442
cases=list(itertools.product([5,1],[1.,.3,10.],[3.141614,3.141615,3.141616]))
for index,(bias,concentration,branch) in enumerate(cases):
 if index%2!=args.worker:continue
 if marker.exists():break
 directory=root/f'b{branch:.6f}-c{concentration}-bias{bias}';directory.mkdir(exist_ok=True)
 output=directory/'best.json';log=directory/'run.log'
 command=['.venv/bin/python','search.py','lll','--input',f'experiments/branch-{branch:.6f}-3.141592.json','--keep-seed','--seconds','30','--target','2.94852698','--goal','2.9485269238','--regularization','.00001','--movement','50','--concentration',str(concentration),'--boundary-bias',str(bias),'--block-size','20','--seed',str(492233+index*1009),'--output',str(output)]
 with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 info={'worker':args.worker,'index':index,'branch':branch,'concentration':concentration,'boundary_bias':bias,'code':result.returncode,'elapsed':time.perf_counter()-start,'output':str(output),'saved':output.exists()}
 if output.exists():
  validation=checked(load(output));info.update(valid=validation.valid,score=validation.score)
  if validation.valid and validation.score<best:
   best=validation.score;response=check_solution(json.loads(output.read_text()));(directory/'api.json').write_text(json.dumps(response,indent=2)+'\n');info['api']=response
   if response.get('valid') and response.get('score',99)<2.9485269238856113:
    marker.write_bytes(output.read_bytes());print(json.dumps(info),flush=True);raise SystemExit('WINNER')
 print(json.dumps(info),flush=True)
