"""Regression checks for protocol isolation, budget forecast and temporal image alignment."""
import sys,json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import visual_study as v

class VisualProtocolTests(unittest.TestCase):
 def test_frozen_protocol_and_observation_inputs(self):
  self.assertEqual(v.OFFSETS,(0,1000000,2000000))
  self.assertEqual(v.SEEDS,(7,17,27))
  for kind in ('bc','diffusion_policy'):
   c=v.config(kind,8,10000,7)
   self.assertEqual(list(c.observation.modalities.obs.low_dim),list(v.LOW_KEYS))
   self.assertEqual(list(c.observation.modalities.obs.rgb),['agentview_image'])
   self.assertEqual(sum(v.SHAPES[k][0] for k in v.LOW_KEYS),9)
   self.assertFalse(c.observation.encoder.rgb.core_kwargs['backbone_kwargs']['pretrained'])
   self.assertIsNone(c.observation.encoder.rgb.obs_randomizer_class)
 def test_forecast_includes_every_candidate_and_noise_repeat(self):
  bc={'train_seconds_per_update':.01,'full400_episode_seconds':2}
  dp={'train_seconds_per_update':.1,'full400_episode_seconds':3}
  expected=(3*10000*.11+3*5*20*5+300*2+900*3+600)*1.25
  self.assertAlmostEqual(v.estimate_seconds(10000,bc,dp),expected)
 def test_reject_overwrite_and_historical_tasks_before_creation(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'new';v.fresh(p)
   with self.assertRaises(FileExistsError):v.fresh(p)
   collision=Path(t)/'collision'
   with patch.object(v,'historical_tasks',return_value={40000}):
    with self.assertRaises(AssertionError):v.tasks(collision,range(40000,40020))
   self.assertFalse(collision.exists())
 def test_cli_refuses_existing_directory_without_writes(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'sentinel').write_text('keep')
   r=subprocess.run([sys.executable,str(v.ROOT/'scripts/visual_study.py'),'prepare','--out',str(p)],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertEqual(sorted(x.name for x in p.iterdir()),['sentinel'])
   self.assertEqual((p/'sentinel').read_text(),'keep')
 def test_image_sequence_boundaries_and_scaling(self):
  from robomimic.utils.dataset import SequenceDataset
  import robomimic.utils.obs_utils as o
  import torch
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'tiny.hdf5'
   with h5py.File(p,'w') as f:
    data=f.create_group('data')
    for ep in range(2):
     g=data.create_group(f'demo_{ep}');g.attrs['num_samples']=3
     g.create_dataset('actions',data=np.full((3,7),ep,dtype='float32'))
     g.create_dataset('obs/agentview_image',data=np.stack([np.full((84,84,3),ep*100+i,dtype='uint8') for i in range(3)]))
   for kind in ('bc','diffusion_policy'):
    c=v.config(kind,8,10000,7)
    ds=SequenceDataset(hdf5_path=str(p),obs_keys=['agentview_image'],action_keys=list(c.train.action_keys),action_config=c.train.action_config,dataset_keys=['actions'],frame_stack=c.train.frame_stack,seq_length=c.train.seq_length,pad_frame_stack=True,pad_seq_length=True,hdf5_cache_mode='low_dim',load_next_obs=False)
    for ep in range(2):
     for t in (0,2):
      item=ds[ds._demo_id_to_start_indices[f'demo_{ep}']+t]
      idx=np.clip(np.arange(t-(c.train.frame_stack-1),t+c.train.seq_length),0,2)
      self.assertTrue(np.array_equal(item['obs']['agentview_image'][:,0,0,0],ep*100+idx))
      self.assertTrue(np.all(item['actions']==ep))
    raw=torch.full((1,84,84,3),255,dtype=torch.uint8)
    processed=o.process_obs(raw,obs_key='agentview_image')
    self.assertEqual(tuple(processed.shape),(1,3,84,84));self.assertTrue(torch.all(processed==1))
    ds.close_and_delete_hdf5_handle()
 def test_selection_tie_and_noise_weighting(self):
  chosen=v.choose_checkpoint([{'step':4000,'successes':10,'episodes':20},{'step':2000,'successes':10,'episodes':20}]);self.assertEqual(chosen['step'],2000)
  results={}
  for kind in ('bc','diffusion_policy'):
   for seed in v.SEEDS:
    offsets=v.OFFSETS if kind=='diffusion_policy' else (0,)
    results[f'{kind}-{seed}']={'records':[{'task':t,'noise_offset':off,'success':kind=='diffusion_policy' and off==0} for t in range(50000,50100) for off in offsets]}
  s=v.summarize(results);self.assertAlmostEqual(s['paired_comparison']['diffusion_minus_bc'],1/3)

if __name__=='__main__':unittest.main()
