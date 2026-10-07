"""Launch one frozen, finite pretraining batch on currently idle GPUs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def main(root,python):
    root=Path(root).resolve()
    if (root/'launch.json').exists():raise FileExistsError('This training batch has already been launched; do not duplicate jobs')
    data=json.loads((root/'data/manifest.json').read_text())
    assert data['status']=='complete' and data['scene_count']==8192
    for p in [root/'data_preflight_audit/dataset_audit.json',root/'smoke_run_v2/audit.json']:
        audit=json.loads(p.read_text())
        assert audit['status']=='passed', (str(p),audit['status'])
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[]
    for line in query.splitlines():
        index,memory,util=[int(v.strip()) for v in line.split(',')]
        if memory<512 and util<5:idle.append(index)
    if len(idle)<4:raise RuntimeError('Need four currently idle GPUs; existing tasks will not be interrupted')
    jobs=[('shared_s41',41,False),('shared_s163',163,False),('shared_s907',907,False),('no_image_s41',41,True)]
    source_files=['scripts/train_mld_shared.py','foodstateedit/material_lineage/learned_mld.py','scripts/prepare_mld_pretraining_data.py']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files}
    plan={'status':'frozen_before_launch','scope':'Initial shared MLD procedural pretraining: cut and rigid lift, not real paired food truth or photographic realism.',
          'seeds':[41,163,907],'no_image_baseline_seed':41,'optimizer_steps_per_job':30000,'warmup_steps':1000,
          'batch_size':24,'tokens_per_training_batch':512,'canonical_grid_size':16,'model_parameters':1095568,
          'state_dim':4,'input_keys':data['inputs'],'training_action_indices':[0,1,2],'ood_action_index':3,
          'transition_training_state':'predicted_posterior_sdf_rgb','transition_evaluation_state':'shared_sampled_joint',
          'source_code_sha256':hashes,'dataset_manifest_sha256':hashlib.sha256((root/'data/manifest.json').read_bytes()).hexdigest(),
          'source_identity_split':data['split_counts'],'real_probe_scope':'16 frozen unpaired photographs, not targets or 3D truth',
          'timeout_policy':'User authorized unrestricted duration; every job has a finite 30000-step horizon; no wall-clock kill.',
          'gpu_snapshot':query,'created_unix':time.time()}
    write(root/'launch_plan.json',plan)
    launched=[]
    for (name,seed,no_image),gpu in zip(jobs,idle):
        out=root/'runs'/name
        if out.exists():raise FileExistsError(str(out))
        out.mkdir(parents=True)
        cmd=[python,str(root/'scripts/train_mld_shared.py'),'--dataset-root',str(root/'data'),'--output-root',str(out),
             '--seed',str(seed),'--steps','30000','--warmup-steps','1000','--batch-size','24','--tokens','512','--checkpoint-every','500']
        if no_image:cmd.append('--no-image')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (root/'logs'/(name+'.log')).open('ab') as log:
            process=subprocess.Popen(cmd,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job={'name':name,'seed':seed,'no_image':no_image,'gpu':gpu,'pid':process.pid,'command':cmd,'output_root':str(out),'started_unix':time.time()}
        launched.append(job);write(root/'launch.json',{'status':'launched','jobs':launched,'plan':str(root/'launch_plan.json')})
        print(json.dumps(job),flush=True)

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
