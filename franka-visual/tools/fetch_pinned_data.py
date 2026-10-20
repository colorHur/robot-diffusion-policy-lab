"""Fetch the archived dataset revision; never replace an existing dataset."""
import argparse, hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    source=json.loads((ROOT/'evidence/dataset-source.json').read_text())
    target=ROOT/'data/lift-ph-low_dim_v15.hdf5'
    if args.verify_only:
        assert target.stat().st_size==source['bytes'] and digest(target)==source['sha256']
        print('Pinned dataset size and SHA256 verified');return
    target.parent.mkdir(exist_ok=True)
    partial=target.with_suffix('.partial')
    if target.exists() or partial.exists():raise FileExistsError('Dataset or partial download exists; inspect it rather than overwrite')
    with urllib.request.urlopen(source['url'],timeout=120) as stream,partial.open('xb') as f:
        while chunk:=stream.read(1024*1024):f.write(chunk)
    assert partial.stat().st_size==source['bytes'] and digest(partial)==source['sha256'],'Downloaded data identity mismatch; partial preserved'
    partial.rename(target)
    print('Pinned dataset downloaded and verified')
if __name__=='__main__':main()
