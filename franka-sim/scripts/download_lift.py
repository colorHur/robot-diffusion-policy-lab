"""Download only the official Lift PH low-dimensional v1.5 dataset."""
import hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'data/lift-ph-low_dim_v15.hdf5';receipt=root/'data/download.json'
if out.exists():raise FileExistsError(out)
repo='robomimic/robomimic_datasets';name='v1.5/lift/ph/low_dim_v15.hdf5'
with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}') as r:revision=json.load(r)['sha']
url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
sha=hashlib.sha256();size=0
with urllib.request.urlopen(url,timeout=120) as stream,out.with_suffix('.partial').open('xb') as target:
    while chunk:=stream.read(1024*1024):target.write(chunk);sha.update(chunk);size+=len(chunk)
out.with_suffix('.partial').rename(out)
value=dict(repo=repo,revision=revision,path=name,url=url,bytes=size,sha256=sha.hexdigest());receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
