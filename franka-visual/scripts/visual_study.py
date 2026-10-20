"""Independent visual Panda Lift runner, derived from the frozen low-dimensional runner.
Original scripts and artifacts are never modified. See configs/VISUAL-STUDY-PLAN.md.
"""
import argparse, copy, hashlib, json, math, os, random, shutil, subprocess, sys, time, traceback, signal
from collections import OrderedDict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE_DATA=ROOT/'data/lift-ph-low_dim_v15.hdf5'
PREP=Path(os.environ.get('VISUAL_PREP', str(ROOT/'runs/visual-prepare-20261020'))).resolve()
DATA=PREP/'visual.hdf5'
DATA_SHA='2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540'
PINS={'robomimic':'d309eaecc18acf4152a830a895a6984b8ac71b05','robosuite':'a071383d53568ab798eb315c0e95357911be922d'}
SEEDS=(7,17,27);OFFSETS=(0,1000000,2000000);LOW_KEYS=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
KEYS=(*LOW_KEYS,'agentview_image')
SHAPES=OrderedDict(zip(KEYS,([3],[4],[2],[3,84,84])))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def read(p):return json.loads(Path(p).read_text())
def fresh(p):Path(p).mkdir(parents=True,exist_ok=False)
def sources():
    names=['scripts/study.py','tests/test_visual_study.py','scripts/visual_study.py','scripts/visual_prepare.py','scripts/visual_audit.py','scripts/render_compat.py','visual-study','configs/VISUAL-STUDY-PLAN.md','requirements-lock.txt']
    return {p:digest(ROOT/p) for p in names}

def verify_inputs(expected=None):
    assert digest(SOURCE_DATA)==DATA_SHA,'source dataset changed'
    for name,pin in PINS.items():
        repo=ROOT/'vendor'/name
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==pin
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip(),'upstream modified'
    if expected:
        for name,h in expected.items():assert digest(ROOT/name)==h,'source changed: '+name
    if DATA.exists():
        receipt=read(PREP/'result.json');assert receipt['passed'] and receipt['visually_verified']
        assert digest(DATA)==receipt['visual_sha256'],'derived data changed'

def historical_tasks():
    ids=set()
    for p in (ROOT/'runs').glob('**/*tasks/index.json'):
        if p.is_relative_to(PREP):continue
        ids.update(map(int,read(p)))
    return ids


def config(kind,batch,updates,seed):
    import robomimic.utils.obs_utils as O
    from robomimic.config import config_factory
    c=config_factory(kind)
    with c.values_unlocked():
        c.train.data=[{'path':str(DATA)}];c.train.batch_size=batch;c.train.seed=seed
        c.train.num_epochs=updates//100;c.train.hdf5_filter_key='train';c.train.hdf5_validation_filter_key='valid'
        c.train.num_data_workers=0;c.train.hdf5_normalize_obs=False;c.train.cuda=True
        c.train.hdf5_load_next_obs=False;c.train.hdf5_cache_mode='low_dim'
        c.observation.modalities.obs.low_dim=list(LOW_KEYS);c.observation.modalities.obs.rgb=['agentview_image']
        c.observation.encoder.rgb.core_class='VisualCore'
        c.observation.encoder.rgb.core_kwargs={'backbone_class':'ResNet18Conv','backbone_kwargs':{'pretrained':False},'pool_class':'SpatialSoftmax','pool_kwargs':{'num_kp':32},'feature_dimension':64}
        c.observation.encoder.rgb.obs_randomizer_class=None
        c.experiment.validate=True;c.experiment.rollout.enabled=False;c.experiment.logging.log_wandb=False;c.experiment.logging.log_tb=False
        c.experiment.epoch_every_n_steps=100
        if kind=='diffusion_policy':c.algo.ddpm.enabled=False;c.algo.ddim.enabled=True;c.algo.ddim.num_inference_timesteps=10
        for k in c.algo.optim_params:c.algo.optim_params[k]['num_train_batches']=100;c.algo.optim_params[k]['num_epochs']=updates//100
    c.lock();O.initialize_obs_utils_with_config(c);return c

def setup():
    import torch
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from render_compat import install
    install()

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def dataset(c,split='train',train_stats=None):
    from robomimic.utils.train_utils import dataset_factory
    if split=='valid':
        assert train_stats is not None
        c=copy.deepcopy(c)
        with c.values_unlocked():c.train.hdf5_cache_mode='low_dim'
    ds=dataset_factory(c,KEYS,filter_by_attribute=split)
    if split=='valid':ds.set_action_normalization_stats(copy.deepcopy(train_stats))
    return ds

def model_for(c):
    import torch
    from robomimic.algo import algo_factory
    return algo_factory(c.algo_name,c,SHAPES.copy(),7,torch.device('cuda'))

def optimizer_steps(payload):
    return sorted({int(x['step']) for o in payload['optimizers'].values() for x in o['state'].values()})

def nested_finite(value):
    import torch
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(nested_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(nested_finite(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True

def save_model(model,c,path,step,stats):
    import torch
    import robomimic.utils.file_utils as F
    model_data=model.serialize();assert optimizer_steps(model_data)==[step] and nested_finite(model_data)
    if c.algo_name=='diffusion_policy':assert model.ema.optimization_step==step and model_data['ema'] is not None
    shape={'ac_dim':7,'all_shapes':SHAPES.copy(),'all_obs_keys':list(KEYS),'use_images':True,'use_depths':False}
    payload={'model':model_data,'config':c.dump(),'algo_name':c.algo_name,'env_metadata':F.get_env_metadata_from_dataset(str(SOURCE_DATA)),
             'shape_metadata':shape,'variable_state':{'step':step,'ema_optimization_step':step if c.algo_name=='diffusion_policy' else None},'action_normalization_stats':stats}
    torch.save(payload,path)

def boundary_audit(out):
    """Check actual upstream dataset output against scalar-clipped episode indices."""
    import h5py
    with h5py.File(DATA,'r',swmr=True,libver='latest') as f:
        train=[x.decode() for x in f['mask/train'][:]];valid=[x.decode() for x in f['mask/valid'][:]]
        assert len(train)==180 and len(valid)==20 and set(train).isdisjoint(valid)
        assert set(train+valid)==set(f['data'])
        sample_count=0;norms={}
        for kind in ('bc','diffusion_policy'):
            c=config(kind,8,10000,7);stats=None
            for split in ('train','valid'):
                ds=dataset(c,split,stats);d=getattr(ds,'datasets',[ds])[0]
                assert set(d.demos)==set(train if split=='train' else valid)
                if split=='train':
                    stats=ds.get_action_normalization_stats()
                    assert np.all(stats['actions']['offset']==0) and np.all(stats['actions']['scale']==1)
                    norms[kind]={k:{n:np.asarray(v).tolist() for n,v in x.items()} for k,x in stats.items()}
                for ep in d.demos:
                    n=len(f['data'][ep]['actions']);base=d._demo_id_to_start_indices[ep]
                    for t in sorted({0,n//2,n-1}):
                        item=d[base+t];idx=np.clip(np.arange(t-(c.train.frame_stack-1),t+c.train.seq_length),0,n-1)
                        assert np.array_equal(item['actions'],f['data'][ep]['actions'][:][idx].astype(np.float32))
                        for k in KEYS:assert np.array_equal(item['obs'][k],f['data'][ep]['obs'][k][:][idx]), (ep,t,k)
                        assert np.isfinite(item['actions']).all() and np.max(np.abs(item['actions']))<=1
                        sample_count+=1
        result={'passed':True,'train_episodes':train,'valid_episodes':valid,'episode_boundary_samples':sample_count,'normalization':'identity; statistics computed from train only','normalization_stats':norms,'dataset_sha256':digest(DATA),'source_dataset_sha256':DATA_SHA}
    write(out,result);return result

def load_policy(path):
    import torch
    import robomimic.utils.file_utils as F
    return F.policy_from_checkpoint(ckpt_path=str(path),device=torch.device('cuda'))

def new_env(c,render=False):
    import robomimic.utils.file_utils as F
    import robomimic.utils.env_utils as E
    metadata=F.get_env_metadata_from_dataset(str(SOURCE_DATA))
    metadata['env_kwargs'].update(camera_names=['agentview'],camera_heights=84,camera_widths=84)
    env=E.create_env_from_metadata(metadata,render=False,render_offscreen=True,use_image_obs=True)
    return E.wrap_env_from_config(env,c)

def state_json(state):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in state.items()}
def state_native(state):return {**state,'states':np.asarray(state['states'],dtype=np.float64)}
def close_env(env):
    while hasattr(env,'env'):env=env.env
    env.close()

def tasks(out,task_ids):
    assert set(task_ids).isdisjoint(historical_tasks()),'historical task collision'
    fresh(out);c=config('bc',8,10000,7);env=new_env(c);index={}
    try:
        for task in task_ids:
            seed_all(task);env.reset();s=env.get_state();env.reset_to(s);s['states']=env.get_state()['states'].copy()
            write(out/f'{task}.json',state_json(s));index[str(task)]=digest(out/f'{task}.json')
    finally:close_env(env)
    historical_signatures=set()
    for p in (ROOT/'runs').glob('**/*tasks/index.json'):
        if p.parent==out:continue
        for task in read(p):
            state=read(p.parent/f'{task}.json')['states']
            historical_signatures.add(hashlib.sha256(np.asarray(state,dtype=np.float64).tobytes()).hexdigest())
    current=[hashlib.sha256(np.asarray(read(out/f'{task}.json')['states'],dtype=np.float64).tobytes()).hexdigest() for task in index]
    assert len(set(current))==len(current) and set(current).isdisjoint(historical_signatures),'physical initial-state collision'
    write(out/'index.json',index);return index

def model_state_hash(model):
    import torch
    h=hashlib.sha256()
    groups={'nets':model.nets.state_dict()}
    if getattr(model,'ema',None) is not None:groups['ema']=model.ema.averaged_model.state_dict()
    for g,group in groups.items():
        for k,v in sorted(group.items()):h.update((g+'/'+k).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def episode(policy,env,initial,task,offset,kind,force_horizon=False):
    import torch
    seed_all(424242+task+offset);policy.start_episode();obs=env.reset_to(state_native(initial))
    assert np.array_equal(env.get_state()['states'],np.asarray(initial['states'])),'initial state restoration mismatch'
    planning_times=[];action_times=[];episode_started=time.perf_counter()
    states=[env.get_state()['states']];actions=[];success=False;error=None;ret=0.;seconds=0.;plan_calls=0;denoiser=0
    target=policy.policy.ema.averaged_model['policy']['noise_pred_net'] if kind=='diffusion_policy' else None
    def hook(*_):
        nonlocal denoiser
        denoiser+=1
    handle=target.register_forward_hook(hook) if target is not None else None
    try:
        for _ in range(400):
            is_plan=kind!='diffusion_policy' or len(policy.policy.action_queue)==0
            if is_plan:plan_calls+=1
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.no_grad():act=policy(ob=obs)
            torch.cuda.synchronize();duration=time.perf_counter()-t;seconds+=duration;action_times.append(duration)
            if is_plan:planning_times.append(duration)
            if np.shape(act)!=(7,) or not np.isfinite(act).all():raise ValueError('nonfinite or invalid action')
            obs,reward,done,_=env.step(act);ret+=float(reward);actions.append(act);states.append(env.get_state()['states'])
            success=success or bool(env.is_success()['task'])
            if (done or success) and not force_horizon:break
    except torch.cuda.OutOfMemoryError:raise
    except Exception as exc:error=repr(exc);success=False
    finally:
        if handle is not None:handle.remove()
    record={'task':task,'noise_offset':offset,'success':success,'steps':len(actions),'return':ret,'error':error,'policy_seconds':seconds,'policy_calls':len(actions)+(1 if error else 0),'planning_calls':plan_calls,'denoiser_evaluations':denoiser,'planning_seconds':planning_times,'action_call_seconds':action_times,'camera_inclusive_wall_seconds':time.perf_counter()-episode_started}
    assert kind!='diffusion_policy' or error is not None or denoiser==10*plan_calls
    return record,np.asarray(states),np.asarray(actions).reshape(-1,7)

def batch_iter(c):
    import torch
    ds=dataset(c);stats=ds.get_action_normalization_stats()
    assert np.all(stats['actions']['offset']==0) and np.all(stats['actions']['scale']==1)
    loader=torch.utils.data.DataLoader(ds,batch_size=c.train.batch_size,shuffle=True,num_workers=0,drop_last=True,generator=torch.Generator().manual_seed(c.train.seed+123))
    def batches():
        while True:yield from loader
    return iter(batches()),stats

def update(model,batch,step):
    b=model.process_batch_for_training(batch);b=model.postprocess_batch_for_training(b,obs_normalization_stats=None)
    info=model.train_on_batch(b,epoch=(step-1)//100+1);model.on_gradient_step();log=model.log_info(info)
    assert all(math.isfinite(float(v)) for v in log.values()),'nonfinite loss or gradient'
    return {k:float(v) for k,v in log.items()}

def preflight_worker(a):
    import torch
    from robomimic.algo import RolloutPolicy
    setup();seed_all(7);fresh(a.out);c=config(a.kind,a.batch,10000,7);batches,stats=batch_iter(c)
    torch.cuda.reset_peak_memory_stats();m=model_for(c);m.set_train();t=time.perf_counter()
    for step in range(1,31):log=update(m,next(batches),step)
    torch.cuda.synchronize();start=time.perf_counter()
    for step in range(31,61):log=update(m,next(batches),step)
    torch.cuda.synchronize();train_seconds=(time.perf_counter()-start)/30;peak=torch.cuda.max_memory_reserved()
    assert peak<=4.5*1024**3,f'peak memory {peak/1024**3:.3f}GiB exceeds 4.5GiB'
    p=a.out/'temporary.pth';save_model(m,c,p,60,stats);saved=m.serialize();online={k:v.detach().cpu().clone() for k,v in saved['nets'].items()};ema={k:v.detach().cpu().clone() for k,v in (saved.get('ema') or {}).items()}
    del saved,m,batches;torch.cuda.empty_cache();policy,ck=load_policy(p)
    assert all(torch.equal(v,policy.policy.nets.state_dict()[k].cpu()) for k,v in online.items())
    if ema:assert all(torch.equal(v,policy.policy.ema.averaged_model.state_dict()[k].cpu()) for k,v in ema.items())
    # Actual architectural probe, not just config fields.
    architecture={'parameters':sum(x.numel() for x in policy.policy.nets.parameters()),'ema_reloaded_exactly':bool(ema),'checkpoint_reloaded_exactly':True}
    if a.kind=='diffusion_policy':
        net=policy.policy.nets['policy']['noise_pred_net'];assert net.diffusion_step_encoder[1].in_features==256
        architecture['unet_parameter_shapes']={k:list(v.shape) for k,v in net.state_dict().items() if 'conv' in k and len(v.shape)==3}
    env=new_env(c,render=True)
    try:
        seed_all(17);env.reset();initial=state_json(env.get_state());before=model_state_hash(policy.policy);t=time.perf_counter()
        rec,_,_=episode(policy,env,initial,17,0,a.kind,force_horizon=True);assert rec['error'] is None
        wall=time.perf_counter()-t;assert rec['steps']==400
        # Multi-reset EGL regression, numerical state must not change.
        from PIL import Image
        for i in range(2):
            env.reset_to(state_native(initial));state=env.get_state()['states'].copy();image=env.render(mode='rgb_array',height=256,width=256,camera_name='agentview')
            assert np.array_equal(state,env.get_state()['states']);Image.fromarray(image).save(a.out/f'render-{i}.png')
        assert model_state_hash(policy.policy)==before
    finally:close_env(env)
    result={'passed':True,'kind':a.kind,'batch':a.batch,'pilot_updates':60,'train_seconds_per_update':train_seconds,'full400_episode_seconds':wall,'peak_reserved_bytes':peak,'architecture':architecture,'pilot_last_log':log,'inference_record':rec,'temporary_checkpoint_sha256':digest(p)}
    write(a.out/'result.json',result);print(json.dumps(result),flush=True)
    # Release-only fix: retain isolated pilot checkpoint and identity receipts.
    assert p.is_file()

def child(root,command,log,deadline=None):
    remaining=deadline-time.time() if deadline else None
    if remaining is not None and remaining<=0:raise TimeoutError('8-hour wall-clock limit reached')
    with Path(log).open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/visual_study.py'),*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)

def estimate_seconds(budget,bc,dp):
    checkpoints=budget//2000
    # Full 400-step episodes, all validation candidates and final test; +reset/io overhead.
    base=3*budget*(bc['train_seconds_per_update']+dp['train_seconds_per_update'])
    base+=3*checkpoints*20*(bc['full400_episode_seconds']+dp['full400_episode_seconds'])
    base+=300*bc['full400_episode_seconds']+900*dp['full400_episode_seconds']
    return (base+600)*1.25

def preflight(a):
    verify_inputs();fresh(a.out);started=time.time()-read(PREP/'machine-time.json')['seconds'];write(a.out/'started.json',{'unix_time':started,'deadline':started+8*3600})
    expected=sources();write(a.out/'source-sha256.json',expected)
    boundary_audit(a.out/'data-audit.json');attempts=[];chosen=None
    for batch in (32,16,8):
        try:
            for kind in ('diffusion_policy','bc'):
                folder=a.out/f'{kind}-batch{batch}'
                child(a.out,['preflight-worker','--out',str(folder),'--kind',kind,'--batch',str(batch)],a.out/f'{kind}-batch{batch}.log',started+8*3600)
            chosen=batch;break
        except subprocess.CalledProcessError as error:
            log=(a.out/f'{kind}-batch{batch}.log').read_text();attempts.append({'batch':batch,'kind':kind,'error':repr(error)})
            # Only memory-cap/OOM failures may trigger the prespecified smaller batch.
            if not any(s in log for s in ('OutOfMemoryError','out of memory','exceeds 4.5GiB')):raise
    if chosen is None:raise RuntimeError('no prescribed batch fits')
    bc=read(a.out/f'bc-batch{chosen}/result.json');dp=read(a.out/f'diffusion_policy-batch{chosen}/result.json')
    estimates={'10000':estimate_seconds(10000,bc,dp)};remaining=started+8*3600-time.time()
    updates=10000 if estimates['10000']<=remaining else None
    verify_inputs(expected)
    result={'prepared':str(PREP),'visual_sha256':digest(DATA),'passed':updates is not None,'started_at':started,'deadline':started+8*3600,'batch':chosen,'updates':updates,'estimates_seconds_with_25pct_margin':estimates,'attempts':attempts,'source_sha256':expected,'dataset_sha256':digest(DATA),'source_dataset_sha256':DATA_SHA,'bc':bc,'diffusion_policy':dp,'message':'resource budget frozen; no quality-based changes' if updates else 'resource limit; formal comparison not started'}
    write(a.out/'result.json',result);print(json.dumps(result,indent=2))

def train_worker(a):
    import torch
    setup();seed_all(a.seed);fresh(a.out);c=config(a.kind,a.batch,a.updates,a.seed);write(a.out/'config.json',json.loads(c.dump()))
    batches,stats=batch_iter(c);model=model_for(c);model.set_train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();history=[]
    with (a.out/'updates.jsonl').open('x') as f:
        for step in range(1,a.updates+1):
            log=update(model,next(batches),step)
            if step%100==0:
                torch.cuda.synchronize();row={'step':step,**log,'elapsed_seconds':time.perf_counter()-start};f.write(json.dumps(row)+'\n');f.flush();history.append(row);print(a.kind,a.seed,step,log,flush=True)
            if step%2000==0:save_model(model,c,a.out/f'step-{step}.pth',step,stats)
    write(a.out/'result.json',{'completed':True,'updates':a.updates,'seconds':time.perf_counter()-start,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'parameters':sum(x.numel() for x in model.nets.parameters()),'checkpoints':{p.name:digest(p) for p in a.out.glob('*.pth')},'final_log':history[-1]})

def eval_worker(a):
    import torch
    setup();fresh(a.out);policy,ck=load_policy(a.checkpoint);c=policy.policy.global_config;env=new_env(c)
    before=model_state_hash(policy.policy);cksha=digest(a.checkpoint);records=[];state_index=read(a.tasks/'index.json');start=time.perf_counter()
    try:
        for task in sorted(map(int,state_index)):
            assert digest(a.tasks/f'{task}.json')==state_index[str(task)]
            initial=read(a.tasks/f'{task}.json')
            for offset in a.offsets:
                rec,states,actions=episode(policy,env,initial,task,offset,a.kind)
                rec['initial_state_sha256']=state_index[str(task)];records.append(rec)
                np.savez_compressed(a.out/f'trajectory-{task}-{offset}.npz',states=states,actions=actions)
                write(a.out/f'record-{task}-{offset}.json',rec)
            print(a.kind,a.checkpoint.name,'task',task,flush=True)
        assert model_state_hash(policy.policy)==before and digest(a.checkpoint)==cksha
    finally:close_env(env)
    write(a.out/'result.json',{'checkpoint':str(a.checkpoint.relative_to(ROOT)),'checkpoint_sha256':cksha,'kind':a.kind,'records':records,'successes':sum(x['success'] for x in records),'episodes':len(records),'parameters_unchanged':True,'seconds':time.perf_counter()-start})

def choose_checkpoint(results):
    return min(results,key=lambda r:(-r['successes']/r['episodes'],r['step']))

def summarize(test_results,bootstrap=5000):
    by_kind={};paired={}
    for kind in ('bc','diffusion_policy'):
        matrix=[];per_seed=[]
        for seed in SEEDS:
            r=test_results[f'{kind}-{seed}']['records'];tasks_sorted=sorted({v['task'] for v in r});expected=1 if kind=='bc' else 3
            vals=[]
            for t in tasks_sorted:
                rows=[x for x in r if x['task']==t];assert len(rows)==expected;vals.append(float(np.mean([x['success'] for x in rows])))
            matrix.append(vals);per_seed.append(float(np.mean(vals)))
        by_kind[kind]={'success_rates_by_training_seed':per_seed,'mean_success_rate':float(np.mean(per_seed)),'task_means':np.mean(matrix,axis=0).tolist()}
    diff=np.array(by_kind['diffusion_policy']['task_means'])-np.array(by_kind['bc']['task_means']);rng=np.random.RandomState(20261020)
    draws=np.mean(diff[rng.randint(0,len(diff),size=(bootstrap,len(diff)))],axis=1)
    paired={'diffusion_minus_bc':float(diff.mean()),'task_cluster_percentile_95_interval':np.percentile(draws,[2.5,97.5]).tolist(),'bootstrap_draws':bootstrap,'bootstrap_seed':20261020,'unique_test_tasks':len(diff),'scope':'conditional on three fixed training seeds and three diffusion noise offsets; not training-population uncertainty'}
    return {'methods':by_kind,'paired_comparison':paired}

def render_examples(root,selection):
    import imageio.v2 as imageio
    from PIL import Image
    fresh(root/'videos');c=config('bc',8,10000,7);env=new_env(c,render=True);examples=[]
    try:
        for kind in ('bc','diffusion_policy'):
            for success in (True,False):
                found=None
                for seed in SEEDS:
                    folder=root/'test'/f'{kind}-{seed}'
                    for p in sorted(folder.glob('record-*.json'),key=lambda p:tuple(map(int,p.stem.split('-')[1:]))):
                        rec=read(p)
                        if rec['success']==success and rec['error'] is None:found=(folder,rec);break
                    if found:break
                if found is None:
                    examples.append({'kind':kind,'success':success,'available':False});continue
                folder,rec=found;task=rec['task'];offset=rec['noise_offset'];initial=read(root/'test-tasks'/f'{task}.json');env.reset_to(state_native(initial))
                with np.load(folder/f'trajectory-{task}-{offset}.npz') as data:states=data['states']
                name=f"{kind}-{'success' if success else 'failure'}";indices=sorted(set(range(0,len(states),2))|{len(states)-1})
                with imageio.get_writer(root/'videos'/f'{name}.mp4',fps=10) as writer:
                    for i in indices:
                        env.reset_to({'states':states[i]});before=env.get_state()['states'].copy();im=env.render(mode='rgb_array',height=384,width=384,camera_name='agentview')
                        assert np.array_equal(before,env.get_state()['states']);writer.append_data(im)
                    Image.fromarray(im).save(root/'videos'/f'{name}.png')
                reader=imageio.get_reader(root/'videos'/f'{name}.mp4');assert reader.count_frames()==len(indices);reader.close()
                examples.append({'kind':kind,'success':success,'available':True,'source_record':str((folder/f'record-{task}-{offset}.json').relative_to(root)),'video':f'videos/{name}.mp4','frames':len(indices),'replay':'saved actual policy trajectory; no additional policy rollout'})
    finally:close_env(env)
    write(root/'videos/index.json',examples)

def study(a):
    setup();pre=read(a.preflight/'result.json');assert pre['passed'];verify_inputs(pre['source_sha256']);assert pre['visual_sha256']==digest(DATA);fresh(a.out)
    deadline=time.time()+8*3600-(read(PREP/'machine-time.json')['seconds']+read(a.preflight/'machine-time.json')['seconds']);batch=pre['batch'];updates=pre['updates'];status={'state':'training','completed':[],'started_at':time.time(),'deadline':deadline};write(a.out/'status.json',status)
    manifest={'prepared':str(PREP),'preflight':str(a.preflight),'preflight_sha256':digest(a.preflight/'result.json'),'visual_sha256':digest(DATA),'batch':batch,'updates':updates,'training_seeds':SEEDS,'validation_tasks':[40000,40019],'test_tasks':[50000,50099],'offsets':OFFSETS,'source_sha256':pre['source_sha256'],'dataset_sha256':digest(DATA),'source_dataset_sha256':DATA_SHA,'deadline':deadline,'selection':'max validation success, ties earliest step','method_scope':'complete official BC versus DP pipelines, not isolated diffusion causal effect'}
    write(a.out/'manifest.json',manifest)
    try:
        for kind in ('bc','diffusion_policy'):
            for seed in SEEDS:
                key=f'{kind}-{seed}';status['current']=key;write(a.out/'status.json',status)
                folder=a.out/'models'/key
                child(a.out,['train-worker','--out',str(folder),'--kind',kind,'--seed',str(seed),'--batch',str(batch),'--updates',str(updates)],a.out/f'train-{key}.log',deadline)
                status['completed'].append(key);write(a.out/'status.json',status)
        verify_inputs(pre['source_sha256']);status['state']='validation';write(a.out/'status.json',status)
        tasks(a.out/'validation-tasks',range(40000,40020));selection={}
        for kind in ('bc','diffusion_policy'):
            for seed in SEEDS:
                key=f'{kind}-{seed}';candidates=[]
                for step in range(2000,updates+1,2000):
                    checkpoint=a.out/'models'/key/f'step-{step}.pth';folder=a.out/'validation'/key/f'step-{step}'
                    child(a.out,['eval-worker','--out',str(folder),'--kind',kind,'--checkpoint',str(checkpoint),'--tasks',str(a.out/'validation-tasks'),'--offsets','0'],a.out/f'validation-{key}-{step}.log',deadline)
                    r=read(folder/'result.json');candidates.append({'step':step,'successes':r['successes'],'episodes':r['episodes'],'checkpoint':str(checkpoint),'sha256':digest(checkpoint)})
                selected=choose_checkpoint(candidates);selection[key]={'selected':selected,'candidates':candidates};write(a.out/'selection-progress.json',selection)
        # Freeze all six selections before instantiating any final test task.
        verify_inputs(pre['source_sha256']);write(a.out/'selection-frozen.json',selection);freeze_sha=digest(a.out/'selection-frozen.json')
        write(a.out/'pretest.json',{'selection_sha256':freeze_sha,'selected_checkpoint_sha256':{k:v['selected']['sha256'] for k,v in selection.items()},'source_sha256':pre['source_sha256'],'dataset_sha256':digest(DATA),'source_dataset_sha256':DATA_SHA,'created_at':time.time()})
        status['state']='test';write(a.out/'status.json',status);tasks(a.out/'test-tasks',range(50000,50100));test_results={}
        for kind in ('bc','diffusion_policy'):
            for seed in SEEDS:
                key=f'{kind}-{seed}';selected=selection[key]['selected'];checkpoint=Path(selected['checkpoint']);assert digest(checkpoint)==selected['sha256']
                offsets=OFFSETS if kind=='diffusion_policy' else (0,)
                child(a.out,['eval-worker','--out',str(a.out/'test'/key),'--kind',kind,'--checkpoint',str(checkpoint),'--tasks',str(a.out/'test-tasks'),'--offsets',*map(str,offsets)],a.out/f'test-{key}.log',deadline)
                test_results[key]=read(a.out/'test'/key/'result.json')
        assert digest(a.out/'selection-frozen.json')==freeze_sha;verify_inputs(pre['source_sha256'])
        write(a.out/'summary.json',summarize(test_results));status['state']='rendering';write(a.out/'status.json',status)
        if time.time()>deadline:raise TimeoutError('deadline before rendering')
        render_examples(a.out,selection)
        write_report(a.out)
        from visual_audit import audit
        audit(a.out)
        if time.time()>deadline:raise TimeoutError('deadline during finalization')
        status['state']='complete';status['finished_at']=time.time();write(a.out/'status.json',status)
        print(json.dumps(read(a.out/'summary.json'),indent=2))
    except BaseException as error:
        status.update(state='incomplete',error=repr(error),finished_at=time.time());write(a.out/'status.json',status);raise


def write_report(root):
    summary=read(root/'summary.json');metrics={}
    for kind in ('bc','diffusion_policy'):
        rows=[];training=[]
        for seed in SEEDS:
            rows.extend(read(root/'test'/f'{kind}-{seed}'/'result.json')['records'])
            training.append(read(root/'models'/f'{kind}-{seed}'/'result.json'))
        plans=[t for row in rows for t in row['planning_seconds']]
        offsets=OFFSETS if kind=='diffusion_policy' else (0,)
        metrics[kind]={'success_rates_by_training_seed':summary['methods'][kind]['success_rates_by_training_seed'],
            'noise_success_rates':{str(seed):{str(o):float(np.mean([r['success'] for r in read(root/'test'/f'{kind}-{seed}'/'result.json')['records'] if r['noise_offset']==o])) for o in offsets} for seed in SEEDS},
            'parameters':[r['parameters'] for r in training],'training_seconds':sum(r['seconds'] for r in training),
            'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in training),'replanning_mean_ms':float(np.mean(plans)*1000),'replanning_p95_ms':float(np.percentile(plans,95)*1000),
            'policy_amortized_ms_per_action':1000*sum(r['policy_seconds'] for r in rows)/sum(r['steps'] for r in rows),'camera_inclusive_episode_wall_seconds':sum(r['camera_inclusive_wall_seconds'] for r in rows),'errors':sum(r['error'] is not None for r in rows)}
    write(root/'metrics.json',metrics)
    lines=['# Visual Panda Lift comparison','', 'RGB agentview 84x84 plus 9D robot proprioception. No object truth input. Official success is cube center > table+4cm, without hold-time acceptance.','', '| Method | Seed success rates | Mean | Train minutes | Replan mean / p95 ms |','|---|---|---|---|---|']
    for k,m in metrics.items():
        rates='/'.join(f'{100*x:.2f}%' for x in m['success_rates_by_training_seed'])
        lines.append(f"| {k} | {rates} | {100*summary['methods'][k]['mean_success_rate']:.2f}% | {m['training_seconds']/60:.2f} | {m['replanning_mean_ms']:.2f} / {m['replanning_p95_ms']:.2f} |")
    lines+=['',json.dumps(summary['paired_comparison'],ensure_ascii=False,indent=2),'','100 independent physical tasks; diffusion noise repeats are nested. Interval conditional on frozen models/noise. Architectures/horizons/compute differ; no isolated diffusion causal claim. Historical low-dimensional results are background only. Camera-inclusive wall times are simulation measurements, not real robot real-time guarantees. See metrics.json, manifest.json and audit.json for evidence.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','preflight','run','audit','preflight-worker','train-worker','eval-worker'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path)
    p.add_argument('--kind',choices=['bc','diffusion_policy']);p.add_argument('--batch',type=int)
    p.add_argument('--updates',type=int);p.add_argument('--seed',type=int);p.add_argument('--checkpoint',type=Path)
    history=p.add_mutually_exclusive_group()
    history.add_argument('--history',type=Path,help='local historical low-dimensional run for saved-state diagnostics')
    history.add_argument('--skip-historical-diagnostics',action='store_true',help='explicitly skip private historical trajectories for a new replication; recorded in preparation receipt')
    p.add_argument('--reviewed-samples',action='store_true',help='record that the 12 prepared sample images were visually inspected before preflight')
    p.add_argument('--tasks',type=Path);p.add_argument('--offsets',type=int,nargs='+',default=[0])
    a=p.parse_args();a.out=a.out.resolve()
    if a.command!='audit' and a.out.exists():raise FileExistsError('refusing existing output: '+str(a.out))
    started=time.monotonic()
    if a.command in ('preflight','run'):
        spent=read(PREP/'machine-time.json')['seconds']
        if a.command=='run':spent+=read(a.preflight/'machine-time.json')['seconds']
        remaining=8*3600-spent
        if remaining<=0:raise TimeoutError('machine budget exhausted')
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('machine budget exhausted')))
        signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        if a.command=='prepare':
            from visual_prepare import prepare
            prepare(a.out,history=a.history,skip_history=a.skip_historical_diagnostics)
        elif a.command=='preflight':
            if a.reviewed_samples:
                result=read(PREP/'result.json');manifest=read(PREP/'data-manifest.json');samples=manifest['samples']
                assert result['passed'] and len(samples)==12
                review={'reviewer':'operator invoking --reviewed-samples','reviewed_at':time.time(),'samples':{x['image']:digest(PREP/x['image']) for x in samples},'notes':'checked camera orientation, cube visibility, early/middle/late progression; occlusion by gripper expected'}
                write(PREP/'visual-review.json',review)
                result.update(visually_verified=True,data_manifest_sha256=digest(PREP/'data-manifest.json'),visual_review_sha256=digest(PREP/'visual-review.json'));write(PREP/'result.json',result)
            preflight(a)
        elif a.command=='run':study(a)
        elif a.command=='preflight-worker':preflight_worker(a)
        elif a.command=='train-worker':train_worker(a)
        elif a.command=='eval-worker':eval_worker(a)
        else:
            from visual_audit import audit
            audit(a.out)
    except BaseException as exc:
        if a.out.exists() and a.command!='audit':
            write(a.out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'machine_seconds':time.monotonic()-started,'formal_complete':False})
            if a.command=='run':
                status=read(a.out/'status.json') if (a.out/'status.json').exists() else {}
                status.update(state='incomplete',error=repr(exc));write(a.out/'status.json',status)
        raise
    finally:
        if a.command in ('prepare','preflight','run') and a.out.exists():write(a.out/'machine-time.json',{'seconds':time.monotonic()-started})
if __name__=='__main__':main()
