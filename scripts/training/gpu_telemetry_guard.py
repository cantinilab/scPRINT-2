from __future__ import annotations
import json,subprocess,time
from pathlib import Path
from typing import Any
try:
 from lightning.pytorch.callbacks import Callback
except ImportError:
 class Callback:pass

def parse_nvidia_smi(text:str)->dict[str,float|int]:
 out={}; count=0
 for line in text.splitlines():
  if not line.strip():continue
  idx,util,used,total,power=[x.strip() for x in line.split(',')]; i=int(idx); count+=1
  out[f'fallback/gpu{i}/utilization']=float(util); out[f'fallback/gpu{i}/memory_used_mb']=float(used); out[f'fallback/gpu{i}/memory_total_mb']=float(total); out[f'fallback/gpu{i}/power_watts']=float(power)
 out['fallback/gpu_count']=count; assert count==2,out; return out
class GPUTelemetryGuard(Callback):
 def __init__(self,receipt_path:str,every_n_steps:int=100):super().__init__();self.path=Path(receipt_path);self.every=every_n_steps
 def capture(self,trainer:Any):
  if not getattr(trainer,'is_global_zero',True):return
  cmd=['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits']; p=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=20); metrics=parse_nvidia_smi(p.stdout); step=int(getattr(trainer,'global_step',0)); logger=getattr(trainer,'logger',None); assert logger not in {None,False}; logger.log_metrics(metrics,step=step); self.path.parent.mkdir(parents=True,exist_ok=True); row={'epoch':time.time(),'step':step,'metrics':metrics}; self.path.open('a').write(json.dumps(row,sort_keys=True)+'\n')
 def on_fit_start(self,trainer:Any,pl_module:Any):self.capture(trainer)
 def on_train_batch_end(self,trainer:Any,pl_module:Any,outputs:Any,batch:Any,batch_idx:int):
  if int(getattr(trainer,'global_step',0))%self.every==0:self.capture(trainer)
