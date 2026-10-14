"""Generate disjoint expert trajectory splits and action chunks."""

from pathlib import Path

import numpy as np

from .env import ReachEnv


def collect(path, episodes, seed, prediction_horizon=4, observation_horizon=2):
    images, states, chunks, masks, episode_ids = [], [], [], [], []
    expert_success = 0
    for episode_id in range(episodes):
        env = ReachEnv(seed + episode_id)
        episode = []
        env.reset()
        for _ in range(env.horizon):
            image, state = env.observe()
            action = env.expert_action()
            episode.append((image, state, action))
            _, success, done = env.step(action)
            if done:
                expert_success += int(success)
                break
        for t, (image, state, _) in enumerate(episode):
            history = [episode[max(0, t - observation_horizon + 1 + j)] for j in range(observation_horizon)]
            future = [
                episode[t + j][2] if t + j < len(episode) else np.zeros(2, dtype=np.float32)
                for j in range(prediction_horizon)
            ]
            valid = [t + j < len(episode) for j in range(prediction_horizon)]
            images.append([frame[0] for frame in history])
            states.append([frame[1] for frame in history])
            chunks.append(future)
            masks.append(valid)
            episode_ids.append(episode_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, images=np.stack(images), states=np.stack(states),
        actions=np.asarray(chunks, dtype=np.float32),
        action_mask=np.asarray(masks, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        observation_horizon=np.asarray(observation_horizon),
        prediction_horizon=np.asarray(prediction_horizon),
    )
    return {"episodes": episodes, "samples": len(images), "expert_success": expert_success}


def load(path):
    with np.load(path) as archive:
        return tuple(archive[key].copy() for key in ("images", "states", "actions", "action_mask"))

