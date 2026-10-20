"""Verify completed smoke artifacts without performing additional rollouts."""
from pathlib import Path
import hashlib,json,re,subprocess
from importlib.metadata import version
import numpy as np,torch,imageio.v2 as imageio
root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
p=next((root/'runs/bc').glob('*/*/last.pth'));c=torch.load(p,map_location='cpu',weights_only=False)
assert c['variable_state']['epoch']==10
steps={int(v['step']) for o in c['model']['optimizers'].values() for v in o['state'].values()};assert steps=={1000}
assert all(torch.isfinite(v).all() for v in c['model']['nets'].values())
config=json.loads(c['config']);assert config['train']['num_epochs']==10 and config['train']['seed']==7 and config['train']['batch_size']==100 and config['experiment']['epoch_every_n_steps']==100
log=(root/'runs/bc-smoke.log').read_text();assert 'BC_SMOKE_COMPLETED' in log and 'Traceback (most recent call last)' not in log
losses=[]
for epoch in range(1,11):
    match=re.search(r'Train Epoch '+str(epoch)+r'\s*(\{.*?\})',log,re.S);assert match;v=json.loads(match[1]);assert all(np.isfinite(x) for x in v.values());losses.append(v['Loss'])
check=read(root/'runs/checkpoint-render-check/result.json');assert check['passed'] and check['checkpoint_sha256']==sha(p)
for r in check['records']:
    seed=r['seed'];folder=root/'runs/checkpoint-render-check'
    with np.load(folder/f'trajectory-seed{seed}.npz') as a:
        assert a['actions'].shape==(r['stats']['Horizon'],7) and len(a['states'])==r['stats']['Horizon'];assert np.isfinite(a['actions']).all() and np.isfinite(a['states']).all()
    video=imageio.get_reader(folder/r['video']);assert video.count_frames()==r['frames'];frame=video.get_data(r['frames']//2);assert frame.shape==(512,512,3);video.close()
probe=read(root/'runs/environment-render-repair/result.json');assert probe['passed'] and probe['dataset_episodes']==200 and probe['dataset_transitions']==9666 and probe['action_dim']==7
assert sha(root/'data/lift-ph-low_dim_v15.hdf5')==read(root/'data/download.json')['sha256']
pins={}
for name,expected in [('robomimic','d309eaecc18acf4152a830a895a6984b8ac71b05'),('robosuite','a071383d53568ab798eb315c0e95357911be922d')]:
    repo=root/'vendor'/name;commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert commit==expected
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=repo,text=True).strip();pins[name]=commit
pip_check=subprocess.run([str(root/'.venv/bin/python'),'-m','pip','check'],text=True,capture_output=True);assert pip_check.returncode==0
out={'passed':True,'scope':'first Panda Lift simulation/BC smoke only; not paper reproduction or true held-out benchmark','dataset_episodes':200,'dataset_transitions':9666,'training_updates':1000,'training_epochs':10,'training_seed':7,'training_epoch_losses':losses,'training_rollouts':{'episodes':4,'successes':2,'source':'upstream training log; original videos corrupted and retained'},'checkpoint_render_check':check,'visual_review':'repaired PNG and success/failure video middle frames inspected; images normal','initial_visual_check_failed':True,'source_pins':pins,'versions':{k:version(k) for k in ['torch','torchvision','numpy','mujoco','robosuite','robomimic','egl_probe','evdev','ziglang','cmake']},'pip_check':pip_check.stdout.strip(),'limitations':['BC only; diffusion not trained yet','one training seed; only 4 initial smoke rollouts and 2 separate visualization checks','low-dimensional input includes simulation object truth','Panda simulation does not establish exact FR3 model or real-robot performance','no robot connection, no ROS or driver changes'],'checkpoint_sha256':sha(p)}
(root/'runs/acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
