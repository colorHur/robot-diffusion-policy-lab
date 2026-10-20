# 环境准备与复现边界

原实验为Ubuntu、Python3.11.17、RTX3060 Laptop 6GB、PyTorch2.11.0+cu128、MuJoCo3.3.7。完整依赖版本在requirements-lock.txt，两个上游源码提交在evidence/provenance.json。原机使用本地轮子与CUDA运行库安装，未改变系统环境。全新机器的安装步骤未实际验证，不将依赖清单称作跨平台保证。

## 准备步骤（在franka-visual目录）

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

## 数据与视觉复现

下载固定原始低维数据后，可从XML/状态生成图像。不提供预训练checkpoint，不能只克隆后立即执行策略。发布版源码与原冻结版分别标记；完整跨机器训练未验证，不承诺比特级CUDA重现。

```bash
.venv/bin/python tools/fetch_pinned_data.py
.venv/bin/python tools/fetch_pinned_data.py --verify-only
.venv/bin/python -m pytest -q tests
# 显式说明不复现私人历史诊断；该状态会写入回执。
./visual-study prepare --skip-historical-diagnostics --out runs/NEW-PREP
export VISUAL_PREP="$PWD/runs/NEW-PREP"
# 先检查data-manifest.json列出的12张样例，确认方向/物体/阶段正常：
./visual-study preflight --reviewed-samples --out runs/NEW-PREFLIGHT
# 仅当passed=true且预测预算通过：
./visual-study run --preflight runs/NEW-PREFLIGHT --out runs/NEW-RUN
./visual-study audit --out runs/NEW-RUN
```

若本机有旧低维完整产物，用prepare --history /path/to/panda-study-20261020替代显式跳过。该目录需包含validation/test、任务索引和用于身份核验的模型；其相对结构应与原实验一致。旧失败诊断依赖原模型路径，推荐将原低维run放在本项目runs下再用默认路径。公开仓库只有失败汇总，没有私人原始轨迹。

归档任务40000–40019与50000–50099均已使用。复建相同ID只能称已知协议复现，不是新未见保留集；改变方法后的新实验必须另定任务范围、预算和协议。所有命令使用新输出目录，拒绝覆盖，不自动恢复或重测最终任务。离线原始记录审计需完整本地产物，只发布摘要不能独立重算。

原实验先做诊断且保存完整本机产物，发布版的显式跳过属于可用性选项，未用于既有成绩。--reviewed-samples是操作者确认已查看图片的记录，不是自动质量判定。正式预算仍固定六模型各10000更新；预测不足则停止，不自动改成6000。无真机连接、环境升级或发布操作。
