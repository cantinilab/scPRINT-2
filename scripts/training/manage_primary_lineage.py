#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

BUDGET_SECONDS=360_000; MAX_SEGMENTS=5

def atomic(path:Path,obj:dict):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def sha256(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(16*1024*1024),b''):h.update(c)
 return h.hexdigest()
def begin_segment(root:Path,active:Path,job_id:str,restart_count:int,group:str,run_id:str,now:float|None=None):
 now=time.time() if now is None else now; ledger_path=root/'lineage.json'
 if restart_count==0:
  assert not root.exists(); assert not active.exists(); root.mkdir(parents=True)
  scprint_commit=os.environ["EXPECTED_SCPRINT_COMMIT"]
  scdataloader_commit=os.environ["EXPECTED_SCDATALOADER_COMMIT"]
  ledger={'version':1,'status':'running','logical_group':group,'task':'t_6cff2510','budget_seconds':BUDGET_SECONDS,'max_segments':MAX_SEGMENTS,'cumulative_elapsed_seconds':0.0,'segments':[],'scprint_execution_commit':scprint_commit,'scdataloader_execution_commit':scdataloader_commit,'ntv3_sha256':'762b474c37b6a911395cb9b87d3795ce876e417a9f5f65a7afe867bca52e964d','random_init':True,'wandb_contract':'one immutable online run per 20h allocation under logical_group'}
 else:
  assert ledger_path.is_file() and active.is_file(); ledger=json.loads(ledger_path.read_text()); assert ledger['status']=='running' and ledger['logical_group']==group; assert len(ledger['segments'])==restart_count; assert ledger['segments'][-1]['status']=='ended'; assert run_id not in {x['wandb_run_id'] for x in ledger['segments']}
 assert restart_count < MAX_SEGMENTS; checkpoint=None
 if restart_count:
  cp=root/f'hpc_ckpt_{restart_count}.ckpt'; assert cp.is_file() and cp.stat().st_size>0; checkpoint=str(cp.resolve()); cp_info={'path':checkpoint,'size':cp.stat().st_size,'sha256':sha256(cp)}
 else: cp_info=None
 seg={'segment':restart_count,'scheduler_job_id':job_id,'wandb_run_id':run_id,'started_at_epoch':now,'status':'open','checkpoint_input':cp_info}
 ledger['segments'].append(seg); atomic(ledger_path,ledger); atomic(active,{'status':'running','root':str(root.resolve()),'logical_group':group,'segment':restart_count,'scheduler_job_id':job_id,'wandb_run_id':run_id,'checkpoint':checkpoint,'updated_at_epoch':now})
 return {'segment':restart_count,'checkpoint':checkpoint,'random_init':restart_count==0,'wandb_run_id':run_id,'logical_group':group}
def end_segment(root:Path,job_id:str,restart_count:int,reason:str,now:float|None=None):
 now=time.time() if now is None else now; p=root/'lineage.json'; ledger=json.loads(p.read_text()); seg=ledger['segments'][-1]; assert seg['segment']==restart_count and seg['scheduler_job_id']==job_id and seg['status']=='open'; elapsed=max(0.0,now-seg['started_at_epoch']); seg.update(status='ended',ended_at_epoch=now,elapsed_seconds=elapsed,reason=reason); ledger['cumulative_elapsed_seconds']+=elapsed; assert ledger['cumulative_elapsed_seconds']<=BUDGET_SECONDS+3600; atomic(p,ledger); return ledger

def finalize_lineage(root:Path,active:Path,now:float|None=None):
 now=time.time() if now is None else now; p=root/'lineage.json'; ledger=json.loads(p.read_text()); assert len(ledger['segments'])==MAX_SEGMENTS and all(x['status']=='ended' for x in ledger['segments']); ledger['status']='terminal_100h_complete'; ledger['ended_at_epoch']=now; atomic(p,ledger); atomic(active,{'status':'terminal_100h_complete','root':str(root.resolve()),'logical_group':ledger['logical_group'],'segments':MAX_SEGMENTS,'cumulative_elapsed_seconds':ledger['cumulative_elapsed_seconds'],'updated_at_epoch':now}); return ledger

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True); b=sub.add_parser('begin'); b.add_argument('root',type=Path); b.add_argument('active',type=Path); b.add_argument('job'); b.add_argument('segment',type=int); b.add_argument('group'); b.add_argument('run_id'); e=sub.add_parser('end'); e.add_argument('root',type=Path); e.add_argument('job'); e.add_argument('segment',type=int); e.add_argument('reason'); f=sub.add_parser('finalize'); f.add_argument('root',type=Path); f.add_argument('active',type=Path); a=p.parse_args()
 if a.cmd=='begin': print(json.dumps(begin_segment(a.root,a.active,a.job,a.segment,a.group,a.run_id),sort_keys=True))
 elif a.cmd=='end': print(json.dumps(end_segment(a.root,a.job,a.segment,a.reason),sort_keys=True))
 else: print(json.dumps(finalize_lineage(a.root,a.active),sort_keys=True))
if __name__=='__main__':main()
