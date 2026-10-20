import json,hashlib,os,argparse
from pathlib import Path
import h5py,numpy as np,imageio.v2 as imageio
from PIL import Image
from render_compat import install
install()
import robomimic.utils.file_utils as F
import robomimic.utils.env_utils as E
import robomimic.utils.obs_utils as O
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'runs/environment-smoke');args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
p=root/'data/lift-ph-low_dim_v15.hdf5'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540'
O.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object'],'rgb':[]}})
meta=F.get_env_metadata_from_dataset(str(p));(out/'env_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
with h5py.File(p,'r') as f:
    demos=sorted(f['data'],key=lambda s:int(s.split('_')[-1]));count=sum(len(f['data'][k]['actions']) for k in demos)
    for k in demos:
        d=f['data'][k];assert d['actions'].shape[1]==7 and np.isfinite(d['actions'][:]).all()
        for key in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']:assert len(d['obs'][key])==len(d['actions']) and np.isfinite(d['obs'][key][:]).all()
    d=f['data'][demos[0]];states=d['states'][:];model=d.attrs['model_file']
    ep_meta=d.attrs.get('ep_meta',None)
env=E.create_env_from_metadata(meta,render=False,render_offscreen=True,use_image_obs=False)
try:
    obs=env.reset();assert all(np.isfinite(v).all() for v in obs.values())
    for _ in range(10):obs,reward,done,info=env.step(np.zeros(7));assert all(np.isfinite(v).all() for v in obs.values())
    before=env.get_state()['states'].copy()
    im=env.render(mode='rgb_array',height=384,width=384,camera_name='agentview');Image.fromarray(im).save(out/'panda-lift.png')
    assert np.array_equal(before,env.get_state()['states']), 'render changed physical state'
    state={'states':states[0],'model':model}
    if ep_meta is not None:state['ep_meta']=ep_meta
    env.reset_to(state)
    with imageio.get_writer(out/'official-demo-state-replay.mp4',fps=10) as writer:
        for i in range(0,len(states),2):
            env.reset_to({'states':states[i]});writer.append_data(env.render(mode='rgb_array',height=384,width=384,camera_name='agentview'))
    result={'passed':True,'dataset_episodes':len(demos),'dataset_transitions':count,'action_dim':env.action_dimension,'observation_shapes':{k:list(v.shape) for k,v in obs.items()},'zero_action_steps':10,'render_shape':list(im.shape),'demo_replay_states':len(states),'demo_video_kind':'recorded simulation state playback, not a learned-policy rollout','environment_metadata':meta,'MUJOCO_GL':os.environ.get('MUJOCO_GL')}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
finally:
    env.env.close()
