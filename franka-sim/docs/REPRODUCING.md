# 环境准备与复现边界

原实验为Ubuntu、Python3.11.17、RTX3060 Laptop 6GB、PyTorch2.11.0+cu128、MuJoCo3.3.7。完整依赖版本在requirements-lock.txt，两个上游源码提交在evidence/provenance.json。原机使用本地轮子与CUDA运行库安装，未改变系统环境。全新机器的安装步骤未实际验证，不将依赖清单称作跨平台保证。

## 准备步骤（在franka-sim目录）

先准备独立Python3.11虚拟环境和GPU驱动/EGL运行库；不在实验室现有控制电脑上直接改环境。以下需要网络和本机编译条件，不属于本次重新执行的实验。

```bash
python3.11 -m venv .venv
mkdir -p vendor
# 新目录运行，避免覆盖已有上游源码。
git clone https://github.com/ARISE-Initiative/robomimic.git vendor/robomimic
git -C vendor/robomimic checkout d309eaecc18acf4152a830a895a6984b8ac71b05
git clone https://github.com/ARISE-Initiative/robosuite.git vendor/robosuite
git -C vendor/robosuite checkout a071383d53568ab798eb315c0e95357911be922d
```

锁文件的editable Git项应安装到上面的vendor路径，runner会校验该目录的HEAD和tracked文件状态。不要直接用pip默认的src目录替代。生成一个临时安装清单，仅替换editable安装位置，保留原锁文件字节：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
text=Path('requirements-lock.txt').read_text()
lines=[]
for line in text.splitlines():
    if line.startswith('-e git+'):
        name=line.split('#egg=',1)[1]
        assert name in ('robomimic','robosuite')
        line='-e ./vendor/'+name
    lines.append(line)
text='\n'.join(lines)+'\n'
Path('requirements-install.txt').write_text(text)
PY
```

安装PyTorch cu128包需要相应wheel源；普通包使用PyPI。若系统已有可用C/C++编译器与make，优先使用已有工具。原机缺少编译工具时使用了锁定版本的Zig/CMake/Ninja和tools下包装器构建egl_probe、evdev，未安装系统包。包装器的唯一发布适配是用FRANKA_PROJECT_ROOT替代个人绝对路径。

```bash
export FRANKA_PROJECT_ROOT="$PWD"
# 只在缺少系统编译器时先装这些构建工具，再启用包装器：
.venv/bin/python -m pip install ziglang==0.17.0 cmake==3.31.10 ninja==1.13.2
export PATH="$PWD/tools:$PWD/.venv/bin:$PATH"
export CC="$PWD/tools/cc" CXX="$PWD/tools/cxx"
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-install.txt
.venv/bin/python -m pip check
```

`requirements-lock.txt`是原环境记录，并非保证所有平台都能直接解析的最小依赖方案。若安装失败，先记录平台/依赖差异，不擅改正式协议或宣称复现完成。

## 数据与运行

正式数据SHA为2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540，使用evidence/dataset-source.json中的固定HF revision。tools/fetch_pinned_data.py是发布时新增的固定版本下载辅助，未参与已完成训练。原scripts/download_lift.py为完整性原样保留，会查询最新revision，**本轮复现不要使用它**。

```bash
.venv/bin/python tools/fetch_pinned_data.py
.venv/bin/python tools/fetch_pinned_data.py --verify-only
.venv/bin/python -m pytest -q tests
./sim preflight --out runs/NEW-PREFLIGHT
./sim study --preflight runs/NEW-PREFLIGHT --out runs/NEW-STUDY
./sim audit --out runs/NEW-STUDY
```

EGL修复由scripts/render_compat.py在runner内安装。预检仍需实际检查渲染，不用文件存在代替画面正常。保持单GPU顺序执行，所有实验使用新目录。推理计时源码analysis/fixed_input_latency.py记录原实验固定路径；若对新实验计时，需要明确修改为新run并另存协议，不追溯冒充原实验。

公开evidence只是摘要，不包含完整原始轨迹；重新执行audit需要新生成的完整run或本机归档run。没有模型下载链接，故不能只克隆后立即播放实时策略；media视频可直接查看。上游和数据的许可证/条款仍按各来源执行，本仓库未重新分发其源码、数据或权重。
