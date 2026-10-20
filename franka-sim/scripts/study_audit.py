"""Independent scalar audit of the frozen comparison and saved trajectories."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit(root):
    root=Path(root);project=Path(__file__).resolve().parents[1];m=read(root/'manifest.json');frozen=read(root/'selection-frozen.json');pretest=read(root/'pretest.json')
    assert sha(root/'selection-frozen.json')==pretest['selection_sha256']
    assert sha(project/'data/lift-ph-low_dim_v15.hdf5')==m['dataset_sha256']
    for name,digest in m['source_sha256'].items():assert sha(project/name)==digest,'source changed: '+name
    validation_index=read(root/'validation-tasks/index.json');test_index=read(root/'test-tasks/index.json')
    assert sorted(map(int,validation_index))==list(range(20000,20020));assert sorted(map(int,test_index))==list(range(30000,30100))
    for group,index in [('validation-tasks',validation_index),('test-tasks',test_index)]:
        for t,digest in index.items():assert sha(root/group/f'{t}.json')==digest
    # No initial physical state is shared between validation and test.
    signature=lambda p:hashlib.sha256(np.asarray(read(p)['states'],dtype=np.float64).tobytes()).hexdigest()
    assert {signature(root/'validation-tasks'/f'{t}.json') for t in validation_index}.isdisjoint({signature(root/'test-tasks'/f'{t}.json') for t in test_index})
    count=0;method_values={};per_seed={};noise_scores={};costs={};selected_steps={}
    for kind in ('bc','diffusion_policy'):
        method_values[kind]=[];per_seed[kind]=[];noise_scores[kind]={};costs[kind]={'episodes':0,'steps':0,'policy_seconds':0.,'planning_calls':0,'denoiser_evaluations':0,'errors':0}
        for seed in (7,17,27):
            key=f'{kind}-{seed}';sel=frozen[key];tr=read(root/'models'/key/'result.json');assert tr['completed'] and tr['updates']==m['updates']
            candidates=[]
            for step in range(2000,m['updates']+1,2000):
                p=root/'models'/key/f'step-{step}.pth';assert sha(p)==tr['checkpoints'][p.name]
                ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['variable_state']['step']==step
                for opt in ck['model']['optimizers'].values():assert {int(v['step']) for v in opt['state'].values()}=={step}
                assert all(torch.isfinite(v).all() for v in ck['model']['nets'].values())
                if kind=='diffusion_policy':
                    assert ck['variable_state']['ema_optimization_step']==step and ck['model']['ema']
                    assert all(torch.isfinite(v).all() for v in ck['model']['ema'].values())
                    cfg=json.loads(ck['config']);assert cfg['algo']['ddim']['enabled'] and not cfg['algo']['ddpm']['enabled'] and cfg['algo']['ddim']['num_inference_timesteps']==10
                cfg=json.loads(ck['config']);assert cfg['train']['batch_size']==m['batch'] and cfg['train']['hdf5_filter_key']=='train'
                result=read(root/'validation'/key/f'step-{step}'/'result.json');assert result['checkpoint_sha256']==sha(p)
                raw=[read(z) for z in sorted((root/'validation'/key/f'step-{step}').glob('record-*.json'))]
                assert len(raw)==20 and {v['task'] for v in raw}==set(range(20000,20020)) and all(v['noise_offset']==0 for v in raw)
                for row in raw:
                    assert row['initial_state_sha256']==validation_index[str(row['task'])]
                    states=np.load(root/'validation'/key/f'step-{step}'/f"trajectory-{row['task']}-0.npz")
                    assert states['states'].shape[0]==row['steps']+1 and states['actions'].shape==(row['steps'],7);states.close()
                successes=sum(v['success'] for v in raw);assert successes==result['successes'] and result['parameters_unchanged']
                candidates.append((successes,step));count+=len(raw);del ck
            chosen=sorted(candidates,key=lambda v:(-v[0],v[1]))[0]
            assert chosen[1]==sel['selected']['step'];selected_steps[key]=chosen[1]
            chosen_path=Path(sel['selected']['checkpoint']);assert sha(chosen_path)==sel['selected']['sha256']==pretest['selected_checkpoint_sha256'][key]
            result=read(root/'test'/key/'result.json');assert result['checkpoint_sha256']==sel['selected']['sha256'] and result['parameters_unchanged']
            offsets=(0,) if kind=='bc' else (0,1000000,2000000);rows=[]
            for task in range(30000,30100):
                for offset in offsets:
                    row=read(root/'test'/key/f'record-{task}-{offset}.json');assert row['task']==task and row['noise_offset']==offset
                    assert row['initial_state_sha256']==test_index[str(task)];assert 0<=row['steps']<=400
                    if row['error']:assert not row['success']
                    else:
                        assert 1<=row['steps']<=400 and row['policy_calls']==row['steps']
                        assert row['planning_calls']==(math.ceil(row['steps']/8) if kind=='diffusion_policy' else row['steps'])
                        assert row['denoiser_evaluations']==(row['planning_calls']*10 if kind=='diffusion_policy' else 0)
                    assert row['policy_seconds']>=0 and math.isfinite(row['policy_seconds'])
                    with np.load(root/'test'/key/f'trajectory-{task}-{offset}.npz') as a:
                        assert a['actions'].shape==(row['steps'],7) and len(a['states'])==row['steps']+1
                        assert np.isfinite(a['states']).all() and np.isfinite(a['actions']).all()
                        assert np.array_equal(a['states'][0],np.asarray(read(root/'test-tasks'/f'{task}.json')['states']))
                    rows.append(row);count+=1
                    c=costs[kind];c['episodes']+=1;c['steps']+=row['steps'];c['policy_seconds']+=row['policy_seconds'];c['planning_calls']+=row['planning_calls'];c['denoiser_evaluations']+=row['denoiser_evaluations'];c['errors']+=int(row['error'] is not None)
            assert sum(v['success'] for v in rows)==result['successes']
            vals=[sum(float(v['success']) for v in rows if v['task']==task)/len(offsets) for task in range(30000,30100)]
            method_values[kind].append(vals);per_seed[kind].append(sum(vals)/100)
            noise_scores[kind][str(seed)]={str(o):sum(v['success'] for v in rows if v['noise_offset']==o)/100 for o in offsets}
    diffs=[]
    for i in range(100):diffs.append(sum(method_values['diffusion_policy'][s][i]-method_values['bc'][s][i] for s in range(3))/3)
    rng=np.random.RandomState(20261020);draws=[]
    for _ in range(5000):draws.append(sum(diffs[j] for j in rng.randint(0,100,size=100))/100)
    summary=read(root/'summary.json');interval=np.percentile(draws,[2.5,97.5]);assert np.allclose(interval,summary['paired_comparison']['task_cluster_percentile_95_interval'],atol=1e-12,rtol=0)
    assert abs(sum(diffs)/100-summary['paired_comparison']['diffusion_minus_bc'])<1e-12
    for k in per_seed:assert np.allclose(per_seed[k],summary['methods'][k]['success_rates_by_training_seed'],atol=1e-12,rtol=0)
    result={'passed':True,'raw_records_checked':count,'unique_test_tasks':100,'selected_steps':selected_steps,'success_rates_by_training_seed':per_seed,'noise_success_rates':noise_scores,'costs':costs,'paired_interval':interval.tolist(),'audit_source_sha256':sha(__file__),'note':'independent arithmetic and artifact audit; not third-party replication'}
    p=root/'audit.json'
    if p.exists():assert read(p)==result,'audit output would change'
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result
