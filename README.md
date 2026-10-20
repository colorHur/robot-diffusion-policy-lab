# Visual reaching: behavior cloning vs action diffusion

A small **simulation-only** robot imitation learning project that runs on a CPU. A two-link planar arm observes a 32×32 RGB view of its arm and target plus joint sine/cosine state. The observation encoder extracts the centroids of two known-color image markers and also runs a small CNN on the full image. An inverse-Jacobian expert supplies demonstrations. Two policies predict four future joint-delta actions from the latest two observations: direct behavior cloning (BC) and an eight-step conditional diffusion model trained with a DDPM loss and sampled with deterministic DDIM. Evaluation executes one or two predicted actions before replanning and reports held-out task success, final distance, and policy latency.

This is a bridge from image diffusion to robot action diffusion. It does **not** establish real-robot, ROS, sim-to-real, or contact-rich manipulation experience.

For the next implementation stages and learning order, see [ROADMAP.md](ROADMAP.md).

## Run

From this directory with Python, PyTorch, NumPy and Pillow installed:

```bash
python -m robot_lab.run data
python -m robot_lab.run train --model bc
python -m robot_lab.run train --model diffusion
python -m robot_lab.run eval
python -m pytest -q --basetemp .pytest_tmp
```

On a CUDA machine, add `--device cuda` to both training and evaluation. Default training uses 2000 optimizer updates per model. Set `--steps` and `--episodes` explicitly when doing a quick smoke run. Default horizons are observation 2, prediction 4, execution 2. To compare replanning frequency, run `python -m robot_lab.run eval --action-horizon 1` and `python -m robot_lab.run eval --action-horizon 2`. Evaluation outputs are saved under `runs/temporal/`. Reproducible aggregate and per-episode results from the measured run are tracked in [results](results/) and explained in [RESULTS.md](RESULTS.md).

## Experiment contract

- Train episodes use seeds 0–239, validation uses 10000–10039, and default test uses 20000–20029. Splits have no trajectory overlap.
- Both models see the same observations and expert action chunks. Their architectures differ by policy head; both use the same visual encoder design.
- The target is visible in the image. The expert uses target coordinates to generate labels; policy inputs contain no target coordinates.
- Samples never cross episode boundaries. Missing past observations repeat the first frame, while missing future actions are masked out of both losses.
- Evaluation executes the requested number of actions from each predicted chunk, then observes again and replans.
- The simulation has a kinematic arm without collisions, obstacles, cameras, or physical contact. It is intended for studying data, action sequences, and closed-loop evaluation.
- Report success rates with the episode count and preferably several training seeds. A single small run is only a smoke test.

The sequence interface follows the observation, prediction, and action horizon design of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). This compact implementation is for learning and does not claim benchmark parity.

## Suggested next extension

After verifying the full experiment, move to a standard manipulation benchmark such as PushT or a LeRobot dataset. Keep the BC comparison and held-out closed-loop evaluation. This project deliberately starts with a dependency-light simulator so the methodology can be checked on a CPU before moving to a heavier benchmark.



## 三维Panda Lift阶段归档（2026-10-20）

[franka-sim/](franka-sim/README.md)包含独立三维仿真实验源码、固定协议、结果、生成视频与后续学习记录。BC89.67%、动作扩散96.67%，100个共同测试任务；详情及局限见该目录。此分支仅新增本轮三维归档，未将其他本地二维未提交实验批量上传。
