import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_audit_wrong_directory_does_not_write(tmp_path):
    (tmp_path/'sentinel').write_text('keep')
    r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/visual_study.py'),'audit','--out',str(tmp_path)],capture_output=True,text=True)
    assert r.returncode!=0
    assert sorted(p.name for p in tmp_path.iterdir())==['sentinel']
def test_missing_private_history_rejected_before_output(tmp_path):
    out=tmp_path/'new'
    r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/visual_study.py'),'prepare','--history',str(tmp_path/'missing'),'--out',str(out)],capture_output=True,text=True)
    assert r.returncode!=0 and '--skip-historical-diagnostics' in r.stderr
    assert not out.exists()
def test_history_and_skip_are_mutually_exclusive(tmp_path):
    r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/visual_study.py'),'prepare','--history','old','--skip-historical-diagnostics','--out',str(tmp_path/'new')],capture_output=True,text=True)
    assert r.returncode!=0 and not (tmp_path/'new').exists()
