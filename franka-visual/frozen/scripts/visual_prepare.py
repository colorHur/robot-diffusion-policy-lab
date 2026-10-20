"""Offline state restoration, diagnostic evidence and derived visual demonstrations."""
import json,time,signal,hashlib
from pathlib import Path
import numpy as np
import h5py
from PIL import Image

def prepare(out):
    import visual_study as v
    import study as old
    out=Path(out);v.fresh(out);started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('preparation machine budget exhausted')))
    signal.alarm(8*3600)
    assert v.digest(v.SOURCE_DATA)==v.DATA_SHA
    v.setup();v.seed_all(7)
    # Register only low dimensional observations for diagnostic restoration.
    c=old.config('bc',8,10000,7);env=old.new_env(c,render=True)
    history=v.ROOT/'runs/panda-study-20261020'
    rules={'schema_version':1,'scope':'observations from saved states, not causal contact/grasp labels',
           'thresholds_m':{'near_eef_cube':0.04,'lateral_displacement':0.03,'small_vertical_displacement':0.02},
           'labels':['no_close_approach','lateral_displacement','limited_vertical_displacement','near_with_closing_command','undetermined_mechanism'],
           'definitions':{'no_close_approach':'minimum eef-to-cube distance > 4cm','lateral_displacement':'maximum XY cube displacement from initial > 3cm','limited_vertical_displacement':'maximum cube z displacement < 2cm','near_with_closing_command':'distance <=4cm and commanded gripper action >0; does not prove contact or grasp','undetermined_mechanism':'physical cause/contact stability is not inferred from these metrics'}}
    def pick(paths,n):return sorted(paths,key=lambda p:str(p.relative_to(history)))[:n]
    val=pick([p for p in (history/'validation').glob('*/*/record-*.json') if not v.read(p)['success']],12)
    failures=pick([p for p in (history/'test').glob('*/record-*.json') if not v.read(p)['success']],10000)
    successes=pick([p for p in (history/'test').glob('*/record-*.json') if v.read(p)['success']],12)
    assert len(val)==12 and len(failures)==61 and len(successes)==12
    (out/'diagnostics').mkdir();(out/'diagnostics/frames').mkdir()
    def classify(paths,group):
        result=[]
        for number,p in enumerate(paths):
            rec=v.read(p);task=rec['task'];offset=rec['noise_offset'];initial=v.read(history/('validation-tasks' if group=='pilot' else 'test-tasks')/f'{task}.json')
            trajectory=p.with_name(f'trajectory-{task}-{offset}.npz')
            with np.load(trajectory) as z:states=z['states'];actions=z['actions']
            env.reset_to(old.state_native(initial));xyz=[];dist=[]
            for state in states:
                obs=env.reset_to({'states':state})
                assert np.array_equal(env.get_state()['states'],state)
                xyz.append(obs['object'][:3]);dist.append(np.linalg.norm(obs['object'][:3]-obs['robot0_eef_pos']))
            xyz=np.asarray(xyz);dist=np.asarray(dist);lateral=np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1);lift=xyz[:,2]-xyz[0,2]
            labels=['undetermined_mechanism']
            if dist.min()>0.04:labels.append('no_close_approach')
            if lateral.max()>0.03:labels.append('lateral_displacement')
            if lift.max()<0.02:labels.append('limited_vertical_displacement')
            if len(actions) and np.any((dist[:-1]<=0.04)&(actions[:,6]>0)):labels.append('near_with_closing_command')
            evidence=[]
            for i in sorted({0,int(dist.argmin()),int(lift.argmax()),len(states)-1}):
                env.reset_to({'states':states[i]});before=env.get_state()['states'].copy()
                im=env.render(mode='rgb_array',height=256,width=256,camera_name='agentview');assert np.array_equal(before,env.get_state()['states'])
                name=f'diagnostics/frames/{group}-{number:03d}-{i:03d}.png';Image.fromarray(im).save(out/name);evidence.append({'frame':i,'image':name})
            model_key=p.parent.name if group!='pilot' else p.parent.parent.name
            model_sha=v.read(history/'selection-frozen.json')[model_key]['selected']['sha256'] if group!='pilot' else v.digest(history/'models'/model_key/(p.parent.name+'.pth'))
            result.append({'source_record':str(p.relative_to(v.ROOT)),'record_sha256':v.digest(p),'trajectory_sha256':v.digest(trajectory),'model_key':model_key,'checkpoint_sha256':model_sha,'task':task,'offset':offset,'success':rec['success'],'labels':labels,'min_eef_cube_distance_m':float(dist.min()),'max_cube_xy_displacement_m':float(lateral.max()),'max_cube_z_displacement_m':float(lift.max()),'evidence':evidence,'hypothesis':'undetermined; commanded closing and proximity do not establish contact/grasp'})
            print('diagnostic',group,number+1,len(paths),flush=True)
        v.write(out/'diagnostics'/f'{group}.json',result);return result
    try:
        pilot=classify(val,'pilot');v.write(out/'diagnostics/taxonomy-frozen.json',{'rules':rules,'pilot_records':[x['source_record'] for x in pilot],'pilot_sha256':v.digest(out/'diagnostics/pilot.json'),'frozen_before_test_labels':True})
        fail=classify(failures,'failures');success=classify(successes,'successes')
        summary={}
        for kind in ('bc','diffusion_policy'):
            rows=[r for r in fail if r['model_key'].startswith(kind+'-')]
            summary[kind]={'failure_replays':len(rows),'unique_failed_tasks':len({r['task'] for r in rows}),'label_counts':{label:sum(label in r['labels'] for r in rows) for label in rules['labels']},'warning':'counts are descriptive; diffusion has 3 noise repeats, BC has 1'}
        v.write(out/'diagnostics/summary.json',summary)
    finally:old.close_env(env)
    # Derived file deliberately excludes object truth, raw simulator states and next_obs.
    c=v.config('bc',8,10000,7);env=v.new_env(c,render=True);samples=[];max_errors={k:0. for k in v.LOW_KEYS};render_seconds=0.;frames=0
    (out/'samples').mkdir();xml_manifest={}
    try:
        with h5py.File(v.SOURCE_DATA,'r') as src,h5py.File(out/'visual.hdf5','x') as dst:
            data=dst.create_group('data');data.attrs.update(dict(src['data'].attrs));src.copy('mask',dst)
            train=[x.decode() for x in src['mask/train'][:]];valid=[x.decode() for x in src['mask/valid'][:]]
            assert len(train)==180 and len(valid)==20 and set(train).isdisjoint(valid)
            assert set(train+valid)==set(src['data'])
            selected=set(sorted(train,key=lambda x:int(x.split('_')[1]))[:2]+sorted(valid,key=lambda x:int(x.split('_')[1]))[:2])
            for name in sorted(src['data'],key=lambda x:int(x.split('_')[1])):
                demo=src['data'][name];states=demo['states'][:];n=len(demo['actions']);assert len(states)==n
                xml=demo.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml
                meta=demo.attrs.get('ep_meta','{}');meta=meta.decode() if isinstance(meta,bytes) else meta
                env.reset_to({'model':xml,'ep_meta':meta,'states':states[0]})
                group=data.create_group(name);group.attrs['num_samples']=n
                for key in ('actions','dones','rewards'):src.copy(demo[key],group,name=key)
                obs_group=group.create_group('obs')
                for key in v.LOW_KEYS:obs_group.create_dataset(key,data=demo['obs'][key][:])
                images=obs_group.create_dataset('agentview_image',(n,84,84,3),dtype='uint8',chunks=(1,84,84,3),compression='gzip',compression_opts=1)
                for i,state in enumerate(states):
                    t=time.monotonic();obs=env.reset_to({'states':state});render_seconds+=time.monotonic()-t
                    assert np.allclose(env.get_state()['states'],state,atol=0,rtol=0)
                    for key in v.LOW_KEYS:
                        error=float(np.max(np.abs(obs[key]-demo['obs'][key][i])));max_errors[key]=max(max_errors[key],error)
                        assert np.allclose(obs[key],demo['obs'][key][i],atol=1e-5,rtol=0),(name,i,key,error,'restored observation mismatch')
                    image=obs['agentview_image'];assert image.shape==(84,84,3) and image.dtype==np.uint8
                    before=env.get_state()['states'].copy();direct=env.render(mode='rgb_array',height=84,width=84,camera_name='agentview')
                    assert np.array_equal(before,env.get_state()['states']) and np.array_equal(image,direct),'render/state/orientation mismatch'
                    images[i]=image;frames+=1
                    if name in selected and i in {0,n//2,n-1}:
                        path=f'samples/{name}-{i}.png';Image.fromarray(image).save(out/path);samples.append({'episode':name,'frame':i,'split':'train' if name in train else 'valid','image':path})
                assert np.array_equal(group['actions'][:],demo['actions'][:])
                xml_manifest[name]={'frames':n,'split':'train' if name in train else 'valid','model_sha256':hashlib.sha256(xml.encode()).hexdigest(),'ep_meta':meta,'frame_mapping':'derived frame i = source states[i], obs[i], actions[i]'}
                dst.flush();print('rendered',name,n,'total',frames,flush=True)
            assert frames==9666
        v.write(out/'data-manifest.json',{'source_sha256':v.DATA_SHA,'camera':'agentview','shape_hwc':[84,84,3],'dtype':'uint8','policy_scaling':'upstream robomimic RGB processing uint8->float32 CHW /255 exactly once','episodes':xml_manifest,'samples':samples,'max_observation_errors':max_errors,'code_sha256':v.sources(),'render_seconds':render_seconds,'frames':frames})
        v.write(out/'result.json',{'passed':True,'visually_verified':False,'machine_seconds':time.monotonic()-started,'visual_sha256':v.digest(out/'visual.hdf5'),'source_sha256':v.DATA_SHA,'frames':frames,'sample_images':samples,'message':'numerical gates passed; sample images require visual review before preflight'})
    finally:v.close_env(env);signal.alarm(0)
