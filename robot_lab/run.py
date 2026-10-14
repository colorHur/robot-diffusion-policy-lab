"""CLI for dataset generation, training, and closed-loop evaluation."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import collect, load
from .env import ReachEnv
from .models import BehaviorCloning, DiffusionPolicy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))


def tensors(path, device):
    images, states, actions, mask = load(path)
    return (
        torch.from_numpy(images).permute(0, 1, 4, 2, 3).float().to(device) / 255,
        torch.from_numpy(states).float().to(device),
        torch.from_numpy(actions).float().to(device) / ReachEnv.action_limit,
        torch.from_numpy(mask).float().to(device),
    )


def make_model(kind, prediction_horizon, observation_horizon, diffusion_steps):
    if kind == "bc":
        return BehaviorCloning(prediction_horizon, observation_horizon)
    if kind == "diffusion":
        return DiffusionPolicy(prediction_horizon, observation_horizon, diffusion_steps)
    raise ValueError(kind)


def bc_loss(prediction, target, mask):
    return ((prediction - target).square() * mask.unsqueeze(-1)).sum() / (
        mask.sum().clamp_min(1) * target.shape[-1]
    )


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    train_data = tensors(args.data / "train.npz", device)
    val_data = tensors(args.data / "val.npz", device)
    prediction_horizon = train_data[2].shape[1]
    observation_horizon = train_data[0].shape[1]
    model = make_model(args.model, prediction_horizon, observation_horizon, args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    record = []
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(train_data[0].shape[0], (args.batch_size,), device=device)
        image, state, action, mask = (tensor[ids] for tensor in train_data)
        loss = model.loss(image, state, action, mask) if args.model == "diffusion" else bc_loss(
            model(image, state), action, mask
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                # Diffusion validation uses fresh noise; this is a denoising loss,
                # while closed-loop task success is the final policy metric.
                values = []
                for start in range(0, val_data[0].shape[0], args.batch_size):
                    batch = [x[start:start + args.batch_size] for x in val_data]
                    value = model.loss(*batch) if args.model == "diffusion" else bc_loss(
                        model(batch[0], batch[1]), batch[2], batch[3]
                    )
                    values.append((float(value), batch[0].shape[0]))
                val_loss = sum(v * n for v, n in values) / sum(n for _, n in values)
            row = {"step": step, "train_loss": round(float(loss), 5),
                   "val_loss": round(val_loss, 5)}
            record.append(row)
            print(json.dumps(row))
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "kind": args.model,
                            "prediction_horizon": prediction_horizon,
                            "observation_horizon": observation_horizon,
                            "diffusion_steps": args.diffusion_steps,
                            "seed": args.seed}, args.out / f"{args.model}.pt")
    report = {"kind": args.model, "seed": args.seed, "updates": args.steps,
              "elapsed_seconds": round(time.perf_counter() - started, 2), "history": record}
    (args.out / f"{args.model}_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", args.out / f"{args.model}.pt")


def load_policy(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = make_model(checkpoint["kind"], checkpoint["prediction_horizon"],
                       checkpoint["observation_horizon"], checkpoint["diffusion_steps"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["kind"]


@torch.no_grad()
def evaluate_policy(model, kind, seeds, device, action_horizon):
    successes, distances, durations = [], [], []
    for seed in seeds:
        env = ReachEnv(int(seed))
        first = env.reset()
        history = [first] * model.observation_horizon
        generator = torch.Generator(device=device).manual_seed(424242 + int(seed))
        elapsed = 0.0
        while env.steps < env.horizon:
            image = torch.from_numpy(np.stack([h[0] for h in history])).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) / 255
            state = torch.from_numpy(np.stack([h[1] for h in history])).unsqueeze(0).to(device)
            started = time.perf_counter()
            chunk = model.sample(image, state, generator) if kind == "diffusion" else model(image, state)
            elapsed += time.perf_counter() - started
            for index in range(min(action_horizon, env.horizon - env.steps)):
                action = chunk[0, index].cpu().numpy() * ReachEnv.action_limit
                observation, success, done = env.step(action)
                history = (history + [observation])[-model.observation_horizon:]
                if done:
                    break
            if done:
                break
        successes.append(int(success))
        distances.append(env.distance())
        durations.append(elapsed / env.steps)
    return {"episodes": len(seeds), "successes": sum(successes),
            "success_rate": round(float(np.mean(successes)), 4),
            "mean_final_distance": round(float(np.mean(distances)), 4),
            "mean_policy_seconds_per_action": round(float(np.mean(durations)), 5),
            "per_episode_success": successes}


def evaluate(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    seeds = list(range(args.test_seed, args.test_seed + args.episodes))
    results = {"task": "2-link planar visual reaching", "test_seeds": seeds,
               "success_radius": ReachEnv.success_radius, "horizon": ReachEnv.horizon,
               "action_horizon": args.action_horizon}
    expert = []
    for seed in seeds:
        env = ReachEnv(seed)
        env.reset()
        for _ in range(env.horizon):
            _, success, done = env.step(env.expert_action())
            if done:
                expert.append(int(success))
                break
    results["expert"] = {"successes": sum(expert), "episodes": len(expert),
                         "success_rate": round(float(np.mean(expert)), 4)}
    for kind in ("bc", "diffusion"):
        path = args.out / f"{kind}.pt"
        if path.exists():
            model, kind = load_policy(path, device)
            if args.action_horizon > model.prediction_horizon:
                raise ValueError("action horizon cannot exceed checkpoint prediction horizon")
            results[kind] = evaluate_policy(model, kind, seeds, device, args.action_horizon)
    output = args.out / f"evaluation_ta{args.action_horizon}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["data", "train", "eval"])
    parser.add_argument("--model", choices=["bc", "diffusion"], default="bc")
    parser.add_argument("--data", type=Path, default=Path("data/temporal"))
    parser.add_argument("--out", type=Path, default=Path("runs/temporal"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=240)
    parser.add_argument("--val-episodes", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--test-seed", type=int, default=20000)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.observation_horizon, args.prediction_horizon, args.action_horizon) < 1:
        parser.error("all horizons must be positive")
    if args.action_horizon > args.prediction_horizon:
        parser.error("action horizon cannot exceed prediction horizon")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.command == "data":
        set_seed(args.seed)
        summary = {
            "train": collect(args.data / "train.npz", args.train_episodes, 0,
                             args.prediction_horizon, args.observation_horizon),
            "val": collect(args.data / "val.npz", args.val_episodes, 10000,
                           args.prediction_horizon, args.observation_horizon),
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

