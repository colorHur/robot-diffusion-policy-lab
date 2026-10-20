"""Post-study timing of fresh planning calls on known smoke task17, no env steps."""
from pathlib import Path
import sys,time,json,hashlib
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study import ROOT,setup,seed_all,load_policy,new_env,close_env,read,write,model_state_hash,digest
root=ROOT/'runs/panda-study-20261020';assert read(root/'status.json')['state']=='complete'
assert time.time()+300<read(root/'manifest.json')['deadline'],'no postprocessing time left'
out=root/'latency';out.mkdir(exist_ok=False);setup();selection=read(root/'selection-frozen.json');rows=[]
for key,value in selection.items():
    path=Path(value['selected']['checkpoint']);assert digest(path)==value['selected']['sha256']
    policy,ck=load_policy(path);c=policy.policy.global_config;env=new_env(c)
    try:
        seed_all(17);env.reset();state=env.get_state();ob=env.reset_to(state);before=model_state_hash(policy.policy);times=[]
        for i in range(55):
            seed_all(424242+i);policy.start_episode();torch.cuda.synchronize();started=time.perf_counter()
            with torch.no_grad():action=policy(ob=ob)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            assert np.shape(action)==(7,) and np.isfinite(action).all()
            if i>=5:times.append(elapsed*1000)
        assert model_state_hash(policy.policy)==before and digest(path)==value['selected']['sha256']
        rows.append({'model':key,'checkpoint_sha256':digest(path),'warmup':5,'timed':50,'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),'milliseconds':times,'scope':'fresh planning calls including observation preprocessing, CUDA-synchronized wall time; smoke task17 fixed observation; no robot/network/env.step'})
    finally:close_env(env)
    del policy,ck;torch.cuda.empty_cache()
write(out/'result.json',{'source_sha256':digest(__file__),'models':rows,'new_training_updates':0,'new_policy_rollout_episodes':0,'note':'each call resets policy action queue, so DP timing measures generation, not queue popping; not end-to-end hardware latency'})
print(json.dumps(rows,indent=2))
