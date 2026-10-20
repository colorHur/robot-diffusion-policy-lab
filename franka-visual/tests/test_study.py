import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study import choose_checkpoint,summarize,fresh,estimate_seconds

def test_selection_tie_chooses_earlier_not_latest():
    rows=[dict(step=10000,successes=14,episodes=20),dict(step=2000,successes=13,episodes=20),dict(step=6000,successes=14,episodes=20)]
    assert choose_checkpoint(rows)['step']==6000

def test_nested_noise_does_not_inflate_independent_tasks():
    data={}
    for kind in ('bc','diffusion_policy'):
        for seed in (7,17,27):
            data[f'{kind}-{seed}']={'records':[dict(task=t,success=(kind=='diffusion_policy' and o==0)) for t in range(100) for o in ((0,) if kind=='bc' else (0,1,2))]}
    result=summarize(data)
    assert result['paired_comparison']['unique_test_tasks']==100
    assert result['paired_comparison']['diffusion_minus_bc']==pytest.approx(1/3)
    assert result['paired_comparison']['task_cluster_percentile_95_interval']==pytest.approx([1/3,1/3])

def test_missing_noise_record_is_rejected():
    data={f'{k}-{s}':{'records':[dict(task=0,success=True)]} for k in ('bc','diffusion_policy') for s in (7,17,27)}
    with pytest.raises(AssertionError):summarize(data)

def test_fresh_does_not_remove_existing_sentinel(tmp_path):
    (tmp_path/'keep').write_text('original')
    with pytest.raises(FileExistsError):fresh(tmp_path)
    assert (tmp_path/'keep').read_text()=='original'

def test_budget_projection_includes_validation_and_test():
    x={'train_seconds_per_update':1.,'full400_episode_seconds':1.}
    # six models, 5 candidate checkpoints, 600 validation + 1200 test, 600s IO reserve.
    assert estimate_seconds(10000,x,x)==pytest.approx((60000+600+1200+600)*1.25)
