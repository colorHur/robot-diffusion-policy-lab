# Panda Lift视觉BC与动作扩散归档

固定agentview 84×84 RGB与9维机器人本体状态，去除10维物体真值。MuJoCo/robosuite仿真，未部署真机。训练与最终测试已完成，本次发布不启动新训练或重测。

| 方法 | seed7 / 17 / 27 | 均值 | 参数 | 三模型训练 |
|---|---|---|---|---|
| 视觉BC | 99% / 98% / 96% | 97.67% | 12.33M | 16.02分钟 |
| 视觉扩散 | 99% / 97.67% / 99.67% | 98.78% | 78.51M | 64.44分钟 |

100个共同物理任务，BC300、扩散900回放。差+1.11个百分点，固定模型/噪声条件下配对任务95%区间[-0.44,+3.00]，不足以确认扩散优势。累计机器运行2.96小时，1800条验证/测试记录本机审计通过，不代表第三方复现。

## 阅读顺序

- [结果及边界](RESULTS.md)
- [开发过程、验收和偏差](docs/DEVELOPMENT.md)
- [准备与复现](docs/REPRODUCING.md)
- [原始冻结源码](frozen/README.md)及[来源身份](evidence/provenance.json)

## 生成视频

| 方法 | 成功 | 失败 |
|---|---|---|
| BC | [播放](media/bc-success.mp4) | [播放](media/bc-failure.mp4) |
| 扩散 | [播放](media/diffusion_policy-success.mp4) | [播放](media/diffusion_policy-failure.mp4) |

均由保存状态重放生成，没有追加策略执行；[索引](evidence/video-index.json)绑定来源记录。公开仓库不包含原始数据、权重和完整轨迹，仅提供汇总证据；完整本机验收需原始产物。

独立入口`./visual-study prepare/preflight/run/audit`。发布版修复预检临时权重保留、审计失败不写已有目录，并提供显式跳过私人历史诊断的选项；这些修复未参与上述成绩。原样源码与发布差异分别保存，不追溯改写结果。完整运行需要Python3.11、GPU/EGL及固定上游，不能只克隆后立即运行策略。
