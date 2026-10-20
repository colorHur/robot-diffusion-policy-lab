import json,hashlib,time
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image
from render_compat import install
install()
import robomimic.utils.file_utils as F
from robomimic.scripts.run_trained_agent import rollout
root=Path(__file__).resolve().parents[1];out=root/'runs/checkpoint-render-check';out.mkdir(exist_ok=False)
p=next((root/'runs/bc').glob('*/*/last.pth'));before=hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2)
policy,ckpt=F.policy_from_checkpoint(ckpt_path=str(p),device=torch.device('cuda'))
env,_=F.env_from_checkpoint(ckpt_dict=ckpt,render=False,render_offscreen=True)
weights={k:v.detach().clone() for k,v in policy.policy.nets.state_dict().items()};records=[]
try:
    for seed in (17,18):
        np.random.seed(seed);torch.manual_seed(seed)
        start=time.perf_counter()
        with imageio.get_writer(out/f'bc-seed{seed}.mp4',fps=10) as writer:
            stats,traj=rollout(policy,env,400,video_writer=writer,video_skip=2,return_obs=True,camera_names=['agentview'])
        assert 1<=stats['Horizon']<=400 and traj['actions'].shape==(stats['Horizon'],7)
        assert np.isfinite(traj['actions']).all() and np.isfinite(traj['states']).all()
        np.savez_compressed(out/f'trajectory-seed{seed}.npz',actions=traj['actions'],states=traj['states'],rewards=traj['rewards'],dones=traj['dones'])
        (out/f'initial-state-seed{seed}.json').write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in traj['initial_state_dict'].items()},indent=2)+'\n')
        physical=env.get_state()['states'].copy()
        im=env.render(mode='rgb_array',height=512,width=512,camera_name='agentview');Image.fromarray(im).save(out/f'final-seed{seed}.png')
        assert np.array_equal(physical,env.get_state()['states'])
        records.append({'seed':seed,'stats':stats,'seconds':time.perf_counter()-start,'video':f'bc-seed{seed}.mp4','frames':(stats['Horizon']-1)//2+1})
    assert all(torch.equal(v,weights[k]) for k,v in policy.policy.nets.state_dict().items())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==before
    result={'passed':True,'purpose':'checkpoint reload and repaired visualization smoke, not a generalization benchmark','checkpoint':str(p.relative_to(root)),'checkpoint_sha256':before,'new_training_updates':0,'model_weights_unchanged':True,'render_preserved_physical_state':True,'records':records}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
finally:env.env.close()
