"""Run fixed short-budget BC; exceptions propagate rather than CLI swallowing them."""
import json,sys
from pathlib import Path
import torch
from robomimic.config import config_factory
from render_compat import install
install()
from robomimic.scripts.train import train
root=Path(__file__).resolve().parents[1];path=root/'configs/bc-smoke.json'
if '--prepare' in sys.argv:
    cfg=json.loads((root/'vendor/robomimic/robomimic/exps/templates/bc.json').read_text())
    cfg['experiment']['name']='bc-lift-smoke-seed7';cfg['experiment']['epoch_every_n_steps']=100
    cfg['experiment']['rollout'].update(n=4,horizon=400,rate=10,warmstart=0)
    cfg['experiment']['save'].update(every_n_epochs=10)
    cfg['experiment']['logging'].update(log_tb=False,log_wandb=False)
    cfg['train'].update(data=[{'path':str(root/'data/lift-ph-low_dim_v15.hdf5')}],output_dir=str(root/'runs/bc'),num_epochs=10,seed=7,cuda=True)
    path.write_text(json.dumps(cfg,indent=2)+'\n');print(path);sys.exit(0)
cfg=config_factory('bc')
with cfg.values_unlocked():cfg.update(json.loads(path.read_text()))
cfg.lock();assert torch.cuda.is_available(),'CUDA required for this recorded smoke run'
destination=Path(cfg.train.output_dir)/cfg.experiment.name
if destination.exists():raise FileExistsError(f'Refusing to overwrite previous experiment: {destination}')
train(cfg,torch.device('cuda'))
print('BC_SMOKE_COMPLETED',flush=True)
