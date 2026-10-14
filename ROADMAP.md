# 从入门仿真到标准机器人学习项目

本文基于 2026-10-14 阅读的官方项目资料，作为本仓库下一阶段的实施与学习清单。每阶段通过验收后再进入下一阶段。

## 开源项目怎么用

| 项目 | 借鉴什么 | 当前决策 |
|---|---|---|
| [原版 Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | 观测时域 `To`、预测时域 `T`、执行时域 `Ta`；序列边界填充；动作归一化；闭环 rollout | **读方法与关键实现**。官方完整复现实验要求 Linux + NVIDIA GPU，不直接把原仓库搬到当前 Windows/CPU 环境。 |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | episode-aware 数据、预处理、策略接口、PushT 训练与评测工具 | **下一阶段主要动手平台**。先跑官方现成流程，再把本项目的对照实验迁移过去。确认版本与 checkpoint 兼容性后固定依赖。 |
| [robomimic](https://robomimic.github.io/docs/) | 专家演示数据、BC 与 Diffusion Policy 基线、实验配置 | **学习评测与基线设计**。暂不同时安装多个仿真框架。 |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) | 更完整的机器人仿真、模仿学习、强化学习 | **后续阶段**。官方完整工作流的推荐硬件显存明显高于当前 6GB，不作为入门前置条件。 |

## 当前项目与标准实现的差距

当前项目已经有专家轨迹、图像与关节状态、BC 与扩散策略、独立测试任务，以及历史观测和动作片段执行。它仍是二维运动学到达任务，缺乏物体接触；手工颜色定位简化了视觉问题；时序版只做了一次训练随机种子。最新结果见 [RESULTS.md](RESULTS.md)，不能声称扩散优于基线。

## 我接下来按这个顺序改

1. **数据与时序正确性（已完成）**：`To`（输入的历史观测数）、`T`（预测动作数）、`Ta`（每轮实际执行数）已分开配置；轨迹边界会填充和掩码，训练与测试按 episode 隔离，测试覆盖首尾边界。
2. **公平对照（进行中）**：BC 和扩散策略已使用相同的观测时域、数据、动作缩放、训练预算与测试任务，并比较了 Ta=1/2。下一步对每个策略至少跑三个训练种子，报告成功数/总数、均值与波动；训练 loss 只用于诊断。
3. **标准任务 PushT**：先在单独的 Python 环境中用 `gym-pusht` 跑通环境和随机/规则动作；再用 LeRobot 的公开数据在同一任务上训练自己的 BC 和 Diffusion Policy。记录模型版本、数据版本、环境版本和可复现命令。先尝试低维状态，再根据显存与耗时增加图像输入。官方较早发布的预训练 checkpoint 与新版 LeRobot 可能存在处理器文件兼容问题，预训练模型评测作为独立兼容性检查，不阻塞数据与自训流程。
4. **作品整理**：提供一张学习曲线、一张成功率对照表、固定测试集的成功与失败轨迹，以及失败分析。只把实际跑过的结果写入简历。

## 你怎么学：四个模块

**1. 机器人与控制基础。** 手算双关节机械臂的正运动学、雅可比矩阵和关节增量到末端位移的关系；解释当前 `ReachEnv` 专家控制器的阻尼最小二乘为何用于接近奇异位形。验收：不看代码写出输入、输出、单位和动作约束。

**2. 模仿学习与时序数据。** 区分状态、观察、动作、轨迹、episode、离线演示、行为克隆和闭环 rollout；能画出 `To/T/Ta` 的时间轴。验收：找出一条轨迹末尾的训练样本，说明哪些动作是填充、哪些会在执行时使用。

**3. 动作扩散。** 从图像扩散已有知识迁移到动作片段：条件信息、加噪对象、噪声预测损失、DDPM/DDIM 采样、动作反归一化。验收：解释为何去噪 loss 下降仍可能让闭环成功率降低，并用本项目结果举例。

**4. 实验与工程。** 固定数据切分、测试种子、检查点选择、依赖版本；看成功率、任务距离、推理延迟与失败轨迹。验收：独立重跑 BC 和扩散策略，写出结果差异及其不确定性。

## 学习资料顺序

1. 本项目的 `env.py`、`data.py`、`models.py`、`run.py` 和 `RESULTS.md`，先能口述完整数据流。
2. [原版 Diffusion Policy 的代码结构说明](https://github.com/real-stanford/diffusion_policy#-codebase-tutorial)和[论文](https://arxiv.org/abs/2303.04137)，重点读时域、归一化和闭环推理。
3. [LeRobot 的 PushT 训练示例](https://github.com/huggingface/lerobot/blob/main/examples/training/train_policy.py)、[评测入口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py)与[数据接口](https://huggingface.co/docs/lerobot/main/api/datasets)。
4. [robomimic 的 Diffusion Policy 教程](https://robomimic.github.io/docs/tutorials/training_diffusion_policy.html)，用来核对实验配置和基线。

## 硬件边界

本机当前可调用的环境是 Python 3.9 + CPU PyTorch；你此前扩散项目记录的是 RTX 3060 Laptop 6GB。下一阶段先在 CPU 上验证数据与评测逻辑；`gym-pusht` 官方示例使用单独的 Python 3.10 环境。之后再尝试 LeRobot/PushT 的低维、小模型训练。原版 Diffusion Policy 官方说明完整仿真复现实验使用 Linux + NVIDIA GPU；Isaac Lab 官方完整安装建议至少 16GB 显存。因此现阶段不需要为 Isaac Lab 升级硬件，也不保证 6GB 能复现论文训练规模。

## Linux 环境与求职准备

当前 Windows 构建号为 22631，符合微软的 WSL 安装系统版本要求。推荐通过 WSL 2 安装 Ubuntu；这不改变原有 Windows 系统，也不会增加显存。安装仍取决于管理员权限与 BIOS 虚拟化设置。安装后用一个独立项目练习 `pwd`、`ls`、`cd`、`mkdir`、`cp`、`grep`/`rg`、`ps`、`nvidia-smi`、`git`、Python 虚拟环境、日志定位，以及从零运行本项目的测试和训练。主要在 Linux 命令行工作时，把项目放在 WSL 的 Linux 文件系统中，以避免跨文件系统读写开销。

求职验收不是“装过 Ubuntu”，而是能在 WSL 中独立搭建环境、启动项目、解释依赖错误、查看 GPU/内存占用，并给出可复现命令。机器人学习/数据工程岗位还需标准任务结果；控制、SLAM 和真机岗位需要额外的 ROS 2、C++、运动学与控制证据。

