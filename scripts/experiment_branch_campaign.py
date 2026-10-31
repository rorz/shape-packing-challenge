"""Run independent reduced-lattice searches across opposed-flat branches."""
import json,subprocess,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import load,checked
from arena import check_solution

start=time.perf_counter();best=2.948526977
branches=list(Path('experiments').glob('branch-*.json'))
branches.sort(key=lambda p: ('3.141592' not in p.stem,p.stem))
for index,branch in enumerate(branches):
 directory=Path('experiments/branch_campaign')/branch.stem;directory.mkdir(parents=True,exist_ok=True)
 output=directory/'best.json';log=directory/'run.log'
 command=['.venv/bin/python','search.py','lll','--input',str(branch),'--keep-seed','--seconds','10','--target','2.94852699','--goal','2.9485269238','--regularization','.00001','--movement','500','--block-size','20','--seed',str(519921+index*1009),'--output',str(output)]
 with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 print(json.dumps({'branch':str(branch),'code':result.returncode,'elapsed':time.perf_counter()-start,'output_exists':output.exists()}),flush=True)
 if not output.exists():continue
 validation=checked(load(output));score=validation.score
 print('LOCAL',validation.valid,score,str(output),flush=True)
 if validation.valid and score<best:
  best=score;response=check_solution(json.loads(output.read_text()));(directory/'api.json').write_text(json.dumps(response,indent=2)+'\n')
  print('API',response,str(output),flush=True)
  if response.get('valid') and response.get('score',99)<2.9485269238856113:
   Path('experiments/campaign_winner.json').write_bytes(output.read_bytes());raise SystemExit('WINNER')
